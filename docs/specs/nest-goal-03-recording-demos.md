# /goal — 03 · Recording / device demos (§3.4) on the frozen Simulator seams

> Self-contained prompt. Open at repo root. Stream: example. Depends on: 02 (merged).

## Session start

1. Create an isolated worktree branch **`nest-goal/03-recording-demos`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — especially the **`02-single-neuron-demos` Lessons** (the 4 reusable Simulator seams: analog `res.trace`/`res.times`, current devices, sweep-by-rebuild, multi-channel generators) and the **`00-validation-harness` Lessons** (`compare_trace`/`compare_distributional`, `CAT_*`). Then read:
   - `docs/nest-status/internal/examples-gap.md` §3.4 and `devices-gap.md` (file-backend P2, detector parity rows).
   - The house style: `examples/nest/{brunel_alpha,one_neuron,testiaf}.py`, `examples/nest/README.md`.
   - The device models you will exercise: `_nest/{multimeter,spike_recorder,correlation_detector,correlomatrix_detector,correlospinmatrix_detector,mip_generator}.py` and `_nest/voltmeter.py` (02's preset).
   - The NEST upstream scripts: `multimeter_file.py`, `recording_demo.py`, `cross_check_mip_corrdet.py`, `correlospinmatrix_detector_two_neuron.py`, `precise_spiking.py`, `plot_weight_matrices.py`, `synapsecollection.py`.
3. Write spec + plan under ``. **Approach-first; wait for approval before code.**

## Goal

Port the §3.4 recording/device demos to `examples/nest/<name>.py` on the `Simulator` API, exercising the recording surface as a parity harness. Most ride 02's seams directly; a few need small additions or hit absent APIs (flagged below — those ship a **skipped placeholder**, per the carve-out rules, not a workaround).

## Scope — by feasibility

**Port now (ride 02 seams + existing devices):**
- `multimeter_file` → **in-memory equivalent** (file backends are `devices-gap.md` P2; do not implement file I/O): record multiple recordables via `voltmeter`/`multimeter` + `res.trace`, assert the per-step `V_m` (and any other recordable) trace vs NEST. This is the recorder-parity demo `index.md` P0 #11 calls for.
- `recording_demo` → the full recording-API tour: multiple recordables (`getattr(pop, name)` per 02's alias map — add aliases if a recordable name is missing), `spike_recorder` + analog together, `res.trace`/`res.rate`/`res.n_events`.
- `cross_check_mip_corrdet` → `mip_generator` (shared-mother Poisson) + `correlation_detector`. Detectors are imperative like multimeter → **tap State / run the detector eagerly post-hoc on recorded spikes** (do not gut them; mirror 02's "bypass, don't gut" rule). Parity: covariance/auto-correlation function vs NEST (`compare_distributional` — note its autocorr/CV path is a **stub** per 00 Lessons; flesh it out here for detector metrics).
- `correlospinmatrix_detector_two_neuron` → binary-neuron correlator; covariance matrix vs NEST.
- `precise_spiking` → the `*_ps` precise-timing variants (`iaf_psc_*_ps`); off-grid spike-time parity (`CAT_E`, sub-step).

**Blocked → skipped placeholder (with a clear `skip("blocked on <API>")`):**
- `plot_weight_matrices` — needs `GetConnections` weight introspection (`network-api-gap.md`). Stub it; note the dependency.
- `synapsecollection` — `SynapseCollection` API is absent (`network-api-gap.md`). Stub it.

## Constraints

- `Simulator` API only; house style of `examples/nest/`. Units in `saiunit`.
- Model/detector bugs fixed in `_nest/<model>.py` (reproducing unit test first, `superpowers:systematic-debugging`), never worked around in the example.
- Detectors and multimeter stay imperative-but-bypassed: **tap `State` / post-process recorded arrays**; nothing imperative inside the `for_loop`.
- The `compare_distributional` autocorr/CV stub (00 Lessons) is yours to complete for the correlation demos — keep it general (00/16 reuse it).

## Validation (via the 00 harness)

`_validation/<name>_test.py`, `requires_nest`. Per demo: multimeter/recording → `compare_trace` + `CAT_B_ALIGNED` (recorder one-step offset, 02 Lesson); detectors → covariance/correlation function within tolerance (`compare_distributional`, documented); precise_spiking → `CAT_E` at sub-step resolution. Blocked demos: skipped placeholder test asserting the skip reason. Add a row per ported demo to `examples/nest/README.md`.

## Edge cases to enumerate + test

Multiple recordables in one multimeter; recording interval ≠ `dt` (set `interval=dt` on the NEST side, 02 Lesson); empty spike train into a detector; `mip_generator` mother-process correlation `p`; precise off-grid times straddling a step boundary; detector window/normalization (`devices-gap.md` P1).

## Definition of done

- [ ] 5 demos ported + runnable; 2 blocked demos as documented skipped placeholders.
- [ ] `compare_distributional` autocorr/CV path implemented + unit-tested.
- [ ] Any recordable-alias / detector seam added in `_network`/`_nest` is TDD-covered + docstringed.
- [ ] Parity tests pass (live NEST) / skip (without); README table updated.
- [ ] >90 % coverage on touched code; Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: the recording-API surface exercised (recordable aliases added, detector post-processing pattern, the completed `compare_distributional` autocorr/CV API that 16 reuses), parity numbers, and the precise list of what's blocked on `GetConnections`/`SynapseCollection`.
2. Verify tests pass / skip; coverage > 90%; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + parity table) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
