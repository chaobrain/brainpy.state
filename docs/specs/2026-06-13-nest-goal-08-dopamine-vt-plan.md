# Plan — 08 · `stdp_dopamine_synapse` + `volume_transmitter`

> Companion to `2026-06-13-nest-goal-08-dopamine-vt-spec.md`. TDD throughout
> (`superpowers:test-driven-development`): red → green → refactor, smallest viable
> step. Each numbered stage is a commit-sized unit.

## Files

| Action | Path | What |
|---|---|---|
| edit | `brainpy_state/_network/_event_plastic.py` | `KernelContext.signals` field; base `_gather_signals()->None`; `update()` threads it; `VoltageCoupledPlasticProj` dual reader (`signal_sources`, relaxed guard, `_gather_signals` override) |
| edit | `brainpy_state/_network/_simulator.py` | `_vt_nodes`; `create()` VT branch; `connect()` VT-bind branch + `vt=` kwarg; `_plastic_proj_cls` signal route; `_connect_pair` signal_sources; phase-0 VT update |
| rewrite | `brainpy_state/_nest/volume_transmitter.py` | JAX-native broadcast node (`n` State, `bind_dopa`, `update`) — retire eager path |
| rewrite | `brainpy_state/_nest/stdp_dopamine_synapse.py` | frozen spec + pure `update` kernel (decision #2a) — retire eager path |
| edit | `brainpy_state/_nest/__init__.py` (+ top-level export) | export rebuilt `volume_transmitter`, `stdp_dopamine_synapse` |
| new | `brainpy_state/_network/_event_plastic_test.py` (extend) | unit-test the `signals` seam in isolation (default-empty; dual reader) |
| new | `brainpy_state/_nest/volume_transmitter_rule_test.py` | NEST-free VT `n(t)` recursion + jit/vmap; replaces eager `volume_transmitter_test.py` |
| new | `brainpy_state/_nest/stdp_dopamine_synapse_rule_test.py` | NEST-free kernel + edge cases + jit/vmap/grad; replaces eager `stdp_dopamine_synapse_test.py` |
| new | `brainpy_state/_nest/_validation/_stdp_dopamine_drive.py` | NEST↔brainpy drive (VT + dopa pool + dopamine proj) |
| new | `brainpy_state/_nest/_validation/volume_transmitter_parity_test.py` | live-NEST `n(t)` parity (upstream-first) |
| new | `brainpy_state/_nest/_validation/stdp_dopamine_synapse_parity_test.py` | live-NEST P0 weight-trajectory + timing sweep |
| delete | `brainpy_state/_nest/volume_transmitter_test.py`, `stdp_dopamine_synapse_test.py` | eager tests replaced (per Part 2.6) |

## Stages

**Stage 1 — the `signals` reader seam (substrate, isolated).**
- Red: extend `_event_plastic_test.py` — a synthetic rule with `signal_reads=('n',)`
  reading `ctx.signals['n']`; assert (i) base `EventPlasticProj` → `ctx.signals is None`;
  (ii) `VoltageCoupledPlasticProj` with `signal_sources={'n':(node,'n')}` broadcasts the
  scalar to all edges; (iii) the guard accepts signal-only (empty `post_state_reads`);
  (iv) clopath-style post-only still works (`signal_reads=()`); (v) read-only.
- Green: implement §2 seam. Keep the `_gather_post_states`/`_gather_signals` symmetry.
- Verify: `clopath_synapse_rule_test.py` + `_event_plastic_test.py` + `_voltage_coupled_plastic_test.py` still pass (primitives #1/#2 untouched).

**Stage 2 — `volume_transmitter` JAX-native node.**
- Red: `volume_transmitter_rule_test.py` — feed a known dopa-spike train to a VT module,
  assert `n(t)` follows `n=n·exp(-dt/tau_n)+count/tau_n`; multi-source sum; `dt` sweep;
  jit/vmap; `deliver_interval` accepted+ignored.
- Green: rewrite `volume_transmitter.py` per §4. Export it.

**Stage 3 — Simulator VT plumbing.**
- Red: a Simulator-level test (`_simulator_plastic_test.py` or new) — `create(volume_transmitter)`,
  `connect(dopa_pool, vt)`, phase-0 update advances `n` from the dopa pool's spikes with the
  expected one-step lag; two VTs independent; `connect(.., vt=)` requires a VT.
- Green: implement §6. Confirm non-dopamine sims unaffected (run an existing plastic sim test).

**Stage 4 — `stdp_dopamine_synapse` spec + rule.**
- Red: `stdp_dopamine_synapse_rule_test.py` — closed-form host reference for the kernel
  (§5d); all §8 edge cases (no-dopa, c=0, clamps, b≠0 sign-flip, simultaneity, broadcast,
  vmap/grad). Hand-build `KernelContext(..., signals={'n':…})`.
- Green: write `stdp_dopamine_synapse.py` per §5. Export it. NumPy docstring citing
  `.h` lines; doctest `Examples`.
- Refactor: factor any shared scalar coercion into `_plastic_base` if it earns reuse.

**Stage 5 — live-NEST drive + VT parity (upstream first).**
- Build `_stdp_dopamine_drive.py`: parallel `nest_*`/`our_*` builders (§7). Resolve the
  dopa-pool→VT wiring on the NEST side (`spike_generator → parrot → volume_transmitter`)
  and `CopyModel(stdp_dopamine_synapse, …, {volume_transmitter: vt, A_plus, …})`.
- `volume_transmitter_parity_test.py`: `n(t)` brainpy vs NEST `GetStatus(conn,'n')` chunked;
  `compare_trace`; lock the VT band **before** touching the synapse parity.

**Stage 6 — dopamine P0 + timing sweep parity.**
- `stdp_dopamine_synapse_parity_test.py`: 3-neuron+1-VT, known pulse pattern, ~5 s weight
  trajectory; assert direction+ordering exact + documented band; eligibility-trace timing
  sweep (§7c). Measure the realised residual; set the final band. If indefensible → §3 (b).

**Stage 7 — coverage, docs, Lessons, git.**
- >90 % on touched code (05 untraced-import + `coverage.start()`+`reload` workaround;
  directory-path `--cov`). Doctest examples verified via `DocTestFinder().find(cls)`.
- Append the CONTEXT.md Lessons entry (signal_reads API, VT-latency scheme + band, VT
  increment/decay mapping, parity numbers, remaining-legacy note).
- Commit (no `Co-Authored-By`) → push → PR → merge → `git switch main && git merge
  --ff-only origin/main` → remove worktree.

## Risks / watch-items
- **`* dt` factor:** verified absent in NEST's event-time form (baked into `expm1`); the
  online form carries `dt` inside `expm1(-taus*dt)` — do **not** double-multiply.
- **One-step `n` lag** (§3b) — the documented divergence; phase-0 VT keeps the dopa stamp
  matching NEST's +1.
- **`tau_minus` on post + dendritic delay** — set identically both sides; disable substrate
  `InputDelay` in the drive (`rule.delay=None`), inject post `d` later (04 convention).
- **VT `deliver_interval`/min_delay triggers** — NEST samples weight at send+trigger times;
  sample brainpy weight at the same times for the comparison.
- **Coverage segfault** under module-form `--cov` — use directory path.
