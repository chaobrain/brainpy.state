# examples/nest/synapsecollection.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""SynapseCollection usage — BLOCKED placeholder.

Port target: NEST's ``synapsecollection.py``. The upstream connects neurons,
retrieves the resulting connections with ``nest.GetConnections()`` (which returns
a ``SynapseCollection``), and reads/sets ``.source``, ``.target``, ``.weight`` to
build and plot weight matrices — across several connection rules
(``one_to_one``, ``pairwise_bernoulli``, ``fixed_total_number``, ``all_to_all``)
with ``nest.random`` ``Parameter``-valued weights drawn at ``Connect`` time.

**Why this is blocked.** The demo's entire subject is the ``SynapseCollection``
object and its ``GetConnections``-based introspection, which ``brainpy.state``
does not provide. It additionally relies on named connection rules and
runtime-evaluated ``Parameter`` weight expressions. See
``docs/nest-status/internal/network-api-gap.md`` §3.8 (``SynapseCollection`` —
missing), §3.1 (``GetConnections`` — missing), §3.9 (connection rules — missing),
and §3.11 (``Parameter`` expressions — missing).

When the planned ``nest_compat`` facade (network-api-gap.md §7, P0/P1) exposes
``SynapseCollection`` introspection, named rules, and ``Parameter`` expressions,
this demo can be ported.

Run:  python examples/nest/synapsecollection.py   # raises NotImplementedError
"""

#: Human-readable reason the port is blocked (asserted by the marker test).
BLOCKED_REASON = (
    "synapsecollection is blocked on the SynapseCollection introspection API: "
    "brainpy.state has no GetConnections/SynapseCollection to read per-edge "
    "source/target/weight, and the demo also needs named connection rules and "
    "runtime Parameter weight expressions "
    "(docs/nest-status/internal/network-api-gap.md §3.8, §3.1, §3.9, §3.11). "
    "Unblocked by the planned nest_compat facade (network-api-gap.md §7)."
)


def main():
    """Raise :class:`NotImplementedError` — the demo is blocked (see module docstring)."""
    raise NotImplementedError(BLOCKED_REASON)


if __name__ == "__main__":
    main()
