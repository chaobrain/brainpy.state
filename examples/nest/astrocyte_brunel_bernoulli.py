# examples/nest/astrocyte_brunel_bernoulli.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""A Brunel network with astrocytes, Bernoulli wiring (NEST port) -- BLOCKED.

This is a documented placeholder, *not* a runnable port. NEST's
``astrocyte_brunel_bernoulli`` builds a balanced random (Brunel) network whose
excitatory population projects to both neurons and astrocytes through
``nest.TripartiteConnect(...)``. The primary neuron->neuron edges use the
``pairwise_bernoulli`` rule, and the astrocyte third-factor edges use the
``third_factor_bernoulli_with_pool`` rule with ``pool_size=10`` and
``pool_type='random'`` (each target neuron draws 10 astrocytes at random). That
pooled tripartite connectivity has no equivalent in the
:class:`~brainpy.state.Simulator` API yet (no astrocyte-pool connectivity rule was
added this cluster -- cluster-15d spec §7), so the network cannot be assembled.

The sibling ``astrocyte_brunel_fixed_indegree`` is identical except its *primary*
rule is ``fixed_indegree``; both share the same astrocyte-pool blocker. What *is*
validated is the per-edge SIC physics: the neuron<->astrocyte SIC loop is
parity-checked against live NEST in ``astrocyte_sic_test.py`` (15d) and shown
end-to-end in ``examples/nest/astrocyte_single.py`` and
``examples/nest/astrocyte_interaction.py``. Only the *connectivity rule* that
assembles the population-scale tripartite network is missing.

See ``docs/nest-status/internal/network-api-gap.md`` and ``examples-gap.md`` §3.8
for the tracked gap.
"""

#: Human-readable reason this port is blocked (asserted verbatim by the marker test
#: ``astrocyte_brunel_test.py``). Mentions the exact NEST call + rule + pool
#: parameters and points at the tracking docs.
BLOCKED_REASON = (
    "astrocyte_brunel_bernoulli is blocked on nest.TripartiteConnect with the "
    "third_factor_bernoulli_with_pool astrocyte-pool connectivity rule "
    "(pool_size=10, pool_type='random'; primary rule pairwise_bernoulli): the "
    "Simulator API has no astrocyte-pool connectivity rule yet (no new connectivity "
    "rule this cluster -- cluster-15d spec §7). The per-edge SIC loop physics is "
    "already validated (15d astrocyte_sic_test.py; demos astrocyte_single / "
    "astrocyte_interaction); only the pooled tripartite connectivity rule is "
    "missing. See docs/nest-status/internal/network-api-gap.md and "
    "examples-gap.md §3.8."
)


def main():
    """Refuse to run: raise :class:`NotImplementedError` with :data:`BLOCKED_REASON`."""
    raise NotImplementedError(BLOCKED_REASON)


if __name__ == "__main__":
    main()
