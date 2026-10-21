# Fix: brainevent >=0.2 TracerArrayConversionError on CSR indptr (merged as #97, bdfe732)

## Spec
- CI (PR #95, #96, scheduled NEST comparison) failed with `TracerArrayConversionError`
  at `brainevent/_csr/hybrid_config.py:307` once brainevent 0.2.1 was installed.
- Cause: `EventPlasticProj` / `_SparseEventMatMul` built `_indptr` with `np.cumsum`
  -> int64 (x64 on). brainevent casts indptr to int32 inside the `CSR` constructor;
  the CSR is built inside `update()` (under jit), so the cast yields a tracer and
  0.2's host readback of indptr (task-workspace sizing) fails.
- Requirement: CSR structure arrays already in brainevent's dtype
  (int32 indices; int32 indptr unless nnz > int32 max -> int64).

## Plan (done)
1. `connectivity.csr_structure` / `csr_indptr_dtype`.
2. Used in `event_plastic.py` and `projections.py`.
3. Regression tests in connectivity/projections/event_plastic tests.
4. `brainevent>=0.2.1`; nest-comparison CI pinned to `nest-simulator=3.9`
   (unpinned resolved 3.8: TripartiteConnect lacks `third_factor_conn_spec`).
