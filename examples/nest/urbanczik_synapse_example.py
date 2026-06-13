# examples/nest/urbanczik_synapse_example.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Urbanczik-Senn dendritic plasticity — BLOCKED placeholder.

Port target: NEST's ``urbanczik_synapse_example.py``. The upstream drives a
two-compartment ``pp_cond_exp_mc_urbanczik`` neuron: a teacher conductance forces
the **somatic** firing rate while plastic ``urbanczik_synapse`` inputs onto the
**dendritic** compartment learn to predict it. Each synapse reads its target's
*dendritic* compartment voltage and the dendritic prediction error (the
difference between the somatic spike train and the rate the dendrite predicts)
and updates so the dendritic potential reproduces the somatically-imposed rate —
the Urbanczik-Senn supervised rule.

**Why this is blocked.** The demo needs two seams the ``Simulator`` API does not
yet provide:

* **A dendritic post-compartment reader on a plastic projection.** The ported
  ``urbanczik_synapse`` still extends the legacy ``NESTSynapse`` base
  (``brainpy_state/_nest/urbanczik_synapse.py``), not the ``Simulator``-API
  plastic projection. ``Simulator.connect`` dispatches plastic synapses to
  ``EventPlasticProj`` / ``VoltageCoupledPlasticProj`` (primitive #2, the
  post-state reader), and that reader exposes only the **somatic** ``V``. The
  Urbanczik rule reads a **dendritic compartment** analog state plus the
  dendritic prediction error, so the synapse must be rebuilt on
  ``VoltageCoupledPlasticProj`` *and* that primitive extended to read a named
  post compartment. See ``docs/nest-status/internal/synapses-plasticity-gap.md``
  §3 (``urbanczik_synapse`` — divergent, "needs ``pp_cond_exp_mc_urbanczik``
  postsynaptic").
* **A validated multi-compartment point-process post.**
  ``pp_cond_exp_mc_urbanczik`` is a two-compartment point-process neuron that is
  still unvalidated against NEST on the ``Simulator`` API (no per-compartment
  state parity), so it cannot yet back a parity test. See
  ``docs/nest-status/internal/neurons-gap.md`` §3 (``pp_cond_exp_mc_urbanczik``
  — unvalidated, "multi-compartment + point-process") and
  ``docs/nest-status/internal/examples-gap.md`` §3.3.

When ``VoltageCoupledPlasticProj`` gains a dendritic post-compartment reader and
``pp_cond_exp_mc_urbanczik`` is validated, this demo can be ported on the
``Simulator`` API exactly like the other four §3.3 plasticity demos.

Run:  python examples/nest/urbanczik_synapse_example.py   # raises NotImplementedError
"""

#: Human-readable reason the port is blocked (asserted by the marker test).
BLOCKED_REASON = (
    "urbanczik_synapse_example is blocked on two Simulator-API seams: (1) the "
    "Urbanczik-Senn rule reads a dendritic post-compartment voltage + prediction "
    "error, but urbanczik_synapse still extends the legacy NESTSynapse base and "
    "VoltageCoupledPlasticProj (the plastic post-state reader) exposes only the "
    "somatic V -- the synapse must be rebuilt on VoltageCoupledPlasticProj with a "
    "named dendritic-compartment reader; and (2) the pp_cond_exp_mc_urbanczik "
    "multi-compartment point-process post is still unvalidated on the Simulator "
    "API (docs/nest-status/internal/synapses-plasticity-gap.md §3, "
    "neurons-gap.md §3, examples-gap.md §3.3)."
)


def main():
    """Raise :class:`NotImplementedError` — the demo is blocked (see module docstring)."""
    raise NotImplementedError(BLOCKED_REASON)


if __name__ == "__main__":
    main()
