# K0 T4 — RNG, connectivity rules, EdgeSet, Dense/CSR storage

_Date: 2026-10-21_ · Parent: [`2026-10-21-k0-kernel-spec.md`](2026-10-21-k0-kernel-spec.md)
§4.4, §4.5, §5 row T4, §6 G4 · Architecture doc §7 (`ConnRule` → `EdgeSet`), §7.2.

## Modules

| module | contents |
|---|---|
| `_kernel/rng.py` | `root_key`, `object_key(seed, uid)`, `derive_key(seed, uid, gid_block, step)`, `population_key`, `connectivity_key`, `post_keys`, `trial_keys` (`independent` / `common`), `UidAllocator` |
| `_kernel/connectivity/edgeset.py` | `EdgeSet` (pytree) over global ids, per-edge weight/delay, `get_connections`, `set` |
| `_kernel/connectivity/rules.py` | `AllToAll`, `OneToOne`, `FixedIndegree`, `FixedTotalNumber`, `PairwiseBernoulli` (+ lower-case aliases); `rule.build(pre_gids, post_gids, key) -> EdgeSet` |
| `_kernel/connectivity/storage.py` | `Dense`, `CSR`, `make_storage`, `payload_path`, `csr_index_dtypes` |

## RNG contract (§4.5)

`derive_key(seed, uid, gid_block, step) = fold_in(fold_in(fold_in(PRNGKey(seed), uid), gid_block), step)`.
`uid` is the deterministic insertion index of the object in the `Network`; **one counter
is shared by populations and projections** (so no two objects share a `uid`).
Connectivity uses the reserved step value `CONNECT_STEP = 2**32 - 1`, so it never
collides with a simulation step. Rules receive the projection key `object_key(seed, uid)`
and derive one key per post neuron: `fold_in(derive_key(.., gid // GID_BLOCK, CONNECT_STEP), gid % GID_BLOCK)`.
The realised `EdgeSet` therefore depends only on `(seed, uid, pre view, post view, rule)`,
never on storage, batch size or evaluation order.

Trials: `trial_keys(key, batch, mode)` folds the trial index in last. Unbatched runs are
trial 0. `independent` gives trial `b` the key `fold_in(key, b)`; `common` gives every
trial trial 0's key. Hence trial 0 is identical for batch 1 and batch B in both modes.

## Rules

All decisions on **global ids**; views must not contain duplicate gids. Concatenated
views are passed as one gid array and sampled as one population. Output edges are in
canonical order (post-major, pre ascending, local indices).

| rule | semantics (NEST 3.10) |
|---|---|
| `all_to_all` | every pair; autapses dropped when `allow_autapses=False`; multapse flag ignored |
| `one_to_one` | equal sizes; self pairs dropped when `allow_autapses=False` |
| `fixed_indegree(K)` | exactly `K` sources per target, uniform over sources (minus self); without multapses distinct sources (Floyd for small `K`, top-k otherwise); `K` > available sources raises |
| `fixed_total_number(N)` | exactly `N` edges, uniform over valid pairs (NEST's rejection of autapses); `allow_multapses=False` raises `NotImplementedError` like NEST |
| `pairwise_bernoulli(p)` | independent Bernoulli per pair, multapse flag ignored; generated in post blocks (bounded memory, never an `(N_pre, N_post)` matrix) |

## Storage

`Dense` holds an `(n_pre, n_post)` weight matrix (multapses summed, non-edges masked so
their gradient is zero); `CSR` holds the per-edge data permuted into row order
(`brainevent.CSR`, duplicates kept). Both: `weight_from_edges`, `weight_to_edges`,
`forward(payload, w, path)`, `forward_etrace(payload, w)`.

- `path='binary'` (`BinaryArray`, event-driven) is allowed only for sources with
  `max_multiplicity == 1` (`payload_path(port)`); counts > 1 use `path='float'`, whose
  VJPs (w.r.t. weights and payload) are exact.
- `forward_etrace` routes through `braintrace.matmul` (Dense, masked by `weight_fn`) or
  `braintrace.sparse_matmul(x, w, sparse_mat=CSR)`.
- Index dtypes: int32 indices; `indptr` int64 above `2**31 - 1` edges (requires x64,
  `RuntimeError` otherwise) — `csr_index_dtypes`.

## Tests (G4 part owned here)

EdgeSet identical Dense vs CSR (round trip), batch 1 vs 4 trial 0, common vs independent;
goal-01 repros (global-id autapses on disjoint/overlapping slices and concatenations,
zero-weight edges and multapses kept by `get_connections`, `set('weight')` alignment,
`fixed_total_number`); VJPs vs dense `jnp` reference with counts {0,1,2,3}; vmap;
braintrace non-zero weight gradients (Dense, CSR); optional live-NEST degree statistics.
