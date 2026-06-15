NEST-Compatible Models — Status & Limitations
==============================================

.. warning::

   **Experimental — In Development.** The NEST-compatible model family in
   ``brainpy.state`` is under active development. This page documents what is
   currently available, what is incomplete, and what users should not yet
   rely on. Parameter names, defaults, and numerical behavior may change
   without notice across 0.0.x releases.


What is the NEST-compatible model family?
-----------------------------------------

The NEST-compatible model family is a set of JAX re-implementations of neuron,
synapse, plasticity, and device models from the
`NEST simulator <https://nest-simulator.readthedocs.io/>`_. The goal is to
let researchers familiar with NEST port models to ``brainpy.state`` with
minimal friction by preserving NEST's parameter names and unit conventions.

The models are built on `brainstate <https://github.com/chaobrain/brainstate>`_,
use `brainunit <https://github.com/chaobrain/brainunit>`_ for physical units, and
compile through JAX to CPU, GPU, and TPU. Where applicable, the dynamics are
differentiable.


What's currently available
--------------------------

NEST-compatible models in ``brainpy.state`` fall into five integration
categories, which differ in how the underlying ODE (if any) is solved:

- **Category A** — AdEx (``aeif_*``), GIF (``gif_*``), GLIF (``glif_*``), and
  conductance-based IAF (``iaf_cond_*``). Full Runge-Kutta-Fehlberg (RKF45)
  integration via ``AdaptiveRungeKuttaStep``.
- **Category B** — Current-based IAF (``iaf_psc_*``). Exact analytical
  propagators; no RKF45 needed.
- **Category C** — Hodgkin-Huxley family (``hh_psc_*``, ``hh_cond_*``,
  ``ht_neuron``). ``AdaptiveRungeKuttaStep`` integration.
- **Category D** — Rate models (``lin_rate_*``, ``tanh_rate_*``,
  ``sigmoid_rate_*``, ``threshold_lin_rate_*``, ``siegert_neuron``).
  Vectorised update patterns.
- **Category E** — Devices (generators, recorders, detectors), static
  synapses, gap junctions, and plasticity rules. No ODE integration.

For the full per-family model lists, see:

- :doc:`/api/nest-base` — base classes (``NESTNeuron``, ``NESTSynapse``,
  ``NESTPlasticity``, ``NESTDevice``)
- :doc:`/api/nest-neurons` — IAF, AdEx, GIF, GLIF, HH, MAT, Izhikevich,
  point-process, binary, rate, and other spiking neuron models
- :doc:`/api/nest-synapses` — static synapses, gap junctions, and special
  connections
- :doc:`/api/nest-plasticity` — STP, STDP, and voltage-based learning rules
- :doc:`/api/nest-devices` — current/spike/Poisson generators, recorders,
  and detectors


Active migration in progress
----------------------------

The NEST-compatible model family is mid-refactor. The items below are
landing across 0.0.x releases:

- **Unit library:** ``brainunit`` → ``brainunit``.
- **ODE integration:** scalar ``np.ndindex`` loops → vectorised
  ``AdaptiveRungeKuttaStep`` with JAX ``while_loop``.
- **State representation:** manual numpy arrays → ``DotDict`` PyTrees.
- **Validation:** ``_to_numpy()``-based comparisons → direct unit-aware
  comparisons gated by ``is_tracer()`` so they only run outside JIT.
- **Error handling:** Python ``raise ValueError`` → JIT-safe
  ``brainstate.transform.jit_error_if()``.

**What this means for users.** APIs you call today may behave subtly
differently after each refactor step lands, and some models may be
temporarily skipped in CI during the transition. Pin your dependency
version (``brainpy.state==0.0.4``) when reproducibility matters.


What users should not rely on yet
---------------------------------

- **Numerical equivalence with NEST.** Most models have not been
  systematically benchmarked against the upstream NEST simulator. Treat
  outputs as plausible but unverified for scientific work.
- **Stable parameter names and defaults.** Names track NEST upstream, but
  defaults and unit conventions may shift during the migration above.
- **Multi-compartment models.** Models with the ``_mc`` suffix are
  particularly experimental — ``iaf_cond_alpha_mc`` remains unvalidated, while
  ``pp_cond_exp_mc_urbanczik`` now has per-compartment live-NEST parity
  (cluster-21).
- **Recording-device fidelity.** Generator and recorder semantics do not yet
  fully match NEST's device model.
- **Plasticity (STDP, STP).** Implementations are present, but learning
  behaviour has not been validated against NEST reference traces.


Recommended usage
-----------------

- Pin your dependency version: ``pip install "brainpy.state==0.0.4"``.
- Validate critical numerical behaviour in a small standalone script that
  locks the parameter values you care about; treat any change in those
  outputs across versions as a possible regression to report.
- Prefer the **BrainPy-style** models for production work and surrogate-
  gradient training — they are stable for the 0.0.x series. See the
  :doc:`/brainpy-guide/index` for the BrainPy-style modeling guide.
- Use NEST-compatible models when you need NEST parameter parity for
  porting or comparison.
- Report numerical mismatches against NEST upstream by opening an issue at
  https://github.com/chaobrain/brainpy.state/issues with a minimal
  reproducer.


Roadmap
-------

High-level direction, deliberately coarse-grained:

- Complete the ``brainunit`` → ``brainunit`` migration across all NEST models.
- Vectorised ODE integration across all Category A and Category C models.
- Per-family validation suites against NEST reference traces.
- Promote individual families from Experimental to Beta as their validation
  suites land.


See also
--------

- :doc:`/api/nest-base`, :doc:`/api/nest-neurons`,
  :doc:`/api/nest-synapses`, :doc:`/api/nest-plasticity`,
  :doc:`/api/nest-devices` — API reference for each NEST family.
- `NEST simulator documentation <https://nest-simulator.readthedocs.io/>`_
  — authoritative reference for upstream model semantics.
- :doc:`/brainpy-guide/index` — the Stable BrainPy-style modeling guide,
  recommended for production use.
