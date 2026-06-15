# examples/nest/astrocyte_small_network.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""A small neuron-astrocyte network (NEST ``astrocyte_small_network``) -- BLOCKED.

This is a documented placeholder, *not* a runnable port. NEST's
``astrocyte_small_network`` wires its neuron populations and astrocytes together
with ``nest.TripartiteConnect(...)``: the neuron->neuron edges use the standard
``pairwise_bernoulli`` rule, while the astrocyte third-factor edges use the
``third_factor_bernoulli_with_pool`` rule with ``pool_size=1`` and
``pool_type='block'`` (each block of postsynaptic neurons draws its astrocyte from
a non-overlapping pool). That pooled tripartite connectivity has no equivalent in
the :class:`~brainpy.state.Simulator` API yet (no astrocyte-pool connectivity rule
was added this cluster -- cluster-15d spec §7), so the network cannot be assembled.

What *is* validated is the per-edge SIC physics this demo would build on: the
neuron<->astrocyte SIC loop is parity-checked against live NEST in
``astrocyte_sic_test.py`` (15d) and demonstrated end-to-end in the runnable ports
``examples/nest/astrocyte_single.py`` and ``examples/nest/astrocyte_interaction.py``.
Only the *connectivity rule* that assembles many such edges is missing.

See ``docs/nest-status/internal/network-api-gap.md`` and ``examples-gap.md`` §3.8
for the tracked gap.
"""

#: Human-readable reason this port is blocked (asserted verbatim by the marker test
#: ``astrocyte_small_network_test.py``). Mentions the exact NEST call + rule + pool
#: parameters and points at the tracking docs.
BLOCKED_REASON = (
    "astrocyte_small_network is blocked on nest.TripartiteConnect with the "
    "third_factor_bernoulli_with_pool astrocyte-pool connectivity rule "
    "(pool_size=1, pool_type='block'): the Simulator API has no astrocyte-pool "
    "connectivity rule yet (no new connectivity rule this cluster -- cluster-15d "
    "spec §7). The per-edge SIC loop physics is already validated (15d "
    "astrocyte_sic_test.py; demos astrocyte_single / astrocyte_interaction); only "
    "the pooled tripartite connectivity rule is missing. See "
    "docs/nest-status/internal/network-api-gap.md and examples-gap.md §3.8."
)


def main():
    """Refuse to run: raise :class:`NotImplementedError` with :data:`BLOCKED_REASON`."""
    raise NotImplementedError(BLOCKED_REASON)


if __name__ == "__main__":
    main()
