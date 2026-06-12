# examples/nest/plot_weight_matrices.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Weight-matrix extraction — BLOCKED placeholder.

Port target: NEST's ``plot_weight_matrices.py``. The upstream builds an
excitatory population ``E`` and an inhibitory population ``I``, connects them
with all four E/I combinations, then extracts the synaptic weight of *every*
realized connection with ``nest.GetConnections(pop_pre, pop_post)`` and
assembles four weight matrices (``EE``, ``EI``, ``IE``, ``II``) for
visualization.

**Why this is blocked.** The port needs post-hoc *connection-weight
introspection*: enumerate the realized synapses between two populations and read
each one's weight keyed by ``(source, target)``. ``brainpy.state`` has no
``GetConnections`` / ``SynapseCollection`` equivalent — a projection exposes
``.weight`` only on a *held* projection object, not as an enumerable connection
set. See ``docs/nest-status/internal/network-api-gap.md`` §3.1 (``GetConnections``
— missing) and §3.8 (``SynapseCollection`` — missing).

When the planned ``nest_compat`` facade (network-api-gap.md §7, P0) exposes
``GetConnections`` weight introspection, this demo can be ported as a
``Simulator`` parity harness like the other ``examples/nest`` demos.

Run:  python examples/nest/plot_weight_matrices.py   # raises NotImplementedError
"""

#: Human-readable reason the port is blocked (asserted by the marker test).
BLOCKED_REASON = (
    "plot_weight_matrices is blocked on connection-weight introspection: "
    "brainpy.state has no GetConnections/SynapseCollection to enumerate the "
    "realized synapses between two populations and read each connection's weight "
    "(docs/nest-status/internal/network-api-gap.md §3.1, §3.8). Unblocked by the "
    "planned nest_compat facade (network-api-gap.md §7, P0)."
)


def main():
    """Raise :class:`NotImplementedError` — the demo is blocked (see module docstring)."""
    raise NotImplementedError(BLOCKED_REASON)


if __name__ == "__main__":
    main()
