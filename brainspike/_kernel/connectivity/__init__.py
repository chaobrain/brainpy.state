# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Connectivity: rules -> :class:`EdgeSet` -> storages (K0 spec §4.4)."""
from brainspike._kernel.connectivity.edgeset import Connections, EdgeSet, as_gids
from brainspike._kernel.connectivity.rules import (
    AllToAll,
    ConnRule,
    FixedIndegree,
    FixedTotalNumber,
    OneToOne,
    PairwiseBernoulli,
    all_to_all,
    fixed_indegree,
    fixed_total_number,
    one_to_one,
    pairwise_bernoulli,
)
from brainspike._kernel.connectivity.storage import (
    CSR,
    PAYLOAD_PATHS,
    STORAGE_KINDS,
    Dense,
    Storage,
    csr_index_dtypes,
    make_storage,
    payload_path,
)

__all__ = [
    'Connections', 'EdgeSet', 'as_gids',
    'AllToAll', 'ConnRule', 'FixedIndegree', 'FixedTotalNumber', 'OneToOne', 'PairwiseBernoulli',
    'all_to_all', 'fixed_indegree', 'fixed_total_number', 'one_to_one', 'pairwise_bernoulli',
    'CSR', 'PAYLOAD_PATHS', 'STORAGE_KINDS', 'Dense', 'Storage', 'csr_index_dtypes',
    'make_storage', 'payload_path',
]
