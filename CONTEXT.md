# CONTEXT.md — NEST-parity goal ledger

> **READ THIS FIRST, every session.** It is the cross-session memory for the
> multi-prompt effort to reach NEST parity across `examples-gap.md` and
> `synapses-plasticity-gap.md`. It is a **committed** file (not gitignored) so it
> survives across worktrees via `main`. Each `/goal` prompt file reads it before
> starting and **appends a Lessons entry** (Part 5) before finishing.

---

## Part 1 — The goal

Two parallel streams, executed cluster-per-file (one worktree → one PR → one
merge per prompt file), in dependency order. Master plan + ordering:
`dev/superpowers/nest-goal-index.md`.

1. **Synapse-rebuild stream.** Take every synapse/plasticity model in
   `docs/nest-status/internal/synapses-plasticity-gap.md` from
   `divergent`/`unvalidated` → `implemented` (documented tolerance + passing
   live-NEST parity), and port the `missing` ones. The current implementations
   are **not JAX-conformant** and are being rebuilt (Part 2).
2. **Example-port stream.** Port every non-`unsupported` example in
   `docs/nest-status/internal/examples-gap.md` §3.1–3.10 to `examples/nest/<name>.py`,
   each as a live-NEST parity harness that drives model fixes.

`unsupported` items are out of scope (MUSIC, SONATA, structural plasticity,
HPC/MPI). `blocked` items (spatial, e-prop) ship a **skipped placeholder** until
their API lands.

## Part 2 — Architecture decisions (AUTHORITATIVE — do not relitigate)

These were settled deliberately. Follow them; if one looks wrong, write the
objection into a Lessons entry, do **not** silently diverge.

1. **The synapse layer is being rebuilt, not patched.** The legacy
   `_nest/*_synapse.py` models emulate NEST's kernel in Python — a
   `defaultdict(list)` event queue (`static_synapse.py:618,1041`), unbounded
   post-history lists (`stdp_synapse.py:514-523,622-637`), host-scalar
   `np`/`math`/`float` math, one object per connection. **None of it runs under
   `jit`/`vmap`/`grad`/`for_loop`.** A separate world (`_network/EventProjection`)
   does static delivery in JAX but never uses these models, so **plastic synapses
   are currently unusable in any real network.**
2. **Rebuild lives in `_network/`, on its own JAX primitives, sharing only
   `brainevent`.** Do **not** build on `_brainpy/`'s `Projection`/`Synapse`/`STP`/
   `align_post_ltp` stack. The substrate is the `brainevent.CSR` event-matmul
   pattern already in `_network/_projections.py:79-107` (`_SparseEventMatMul`),
   plus `InputDelay` for delays.
3. **Division of responsibility.** `_network/` owns the **compute primitive**
   (CSR edge layout, per-edge weight `State`, per-edge/per-neuron trace `State`,
   delay, the event matmul, the update loop). `_nest/*_synapse.py` owns the
   **NEST-faithful spec** (parameter names/defaults/units/init-validation) **+ a
   pure, vectorized `update(state, pre_spike, post_spike, dt) -> state` rule
   kernel** the primitive calls inside the jitted loop. No queues, no history
   lists, no host scalars in the hot path.
4. **NEST fidelity wins.** Where NEST semantics and a brainpy.state convention
   conflict (e.g. NEST's `tau_minus`-on-post vs. trace-on-synapse), the NEST
   adapter overrides and the delta is documented (docstring + `CONTEXT.md`).
5. **Typed family of three primitives** over one shared substrate (CSR edges +
   `InputDelay` + rule-declared `State` allocation):
   - **`EventPlasticProj`** — pre-spike in → weighted event out; per-edge weight +
     per-edge aux State (e.g. STP `u,x`) + per-pre/post-neuron trace State, the
     **rule declares exactly which State it needs**. Per-neuron online traces
     reproduce NEST all-to-all STDP; `stdp_nn_*` declare per-edge/reset state.
     Covers ~19 models (static, STP, STDP core+NN, vogels, jonke, ht, bernoulli,
     cont_delay).
   - **`VoltageCoupledPlasticProj`** — extends the above with a **post-state
     reader** (rule declares which post variables it samples each step). Covers
     `clopath_synapse`, `urbanczik_synapse`, `stdp_dopamine_synapse` (+
     `volume_transmitter` broadcast).
   - **`ContinuousCoupledProj`** — bidirectional/instantaneous + waveform
     relaxation. Covers `gap_junction`, `rate_connection_*`, `diffusion_connection`,
     `sic_connection`. **This is a co-design with the rate *neurons*** (see Part 3)
     and **gets its own grilling** before implementation.
6. **Imperative path is deleted.** The `syn.send()`/`record_post_spike()`/event-
   queue/`set()/get()` machinery is removed (not kept as a shim). The ~28 eager
   `*_test.py` files are **replaced** by vectorized network-level parity tests.
7. **Integration with the Simulator.** Plastic projections read pre **and** post
   spikes from the Simulator's `_SpikeHolder` (post spikes drive K⁻). Weight
   recording reuses the analog-recording (State-tap) mechanism, not an imperative
   hook.

## Part 3 — Repo reality map

- **JAX-native already (safe to build on):** spiking neurons `iaf_*`/`aeif_*`
  (Brunel runs them in `transform.for_loop`, `_network/_simulator.py:268`); the
  `Simulator` API (`_network/`: `Simulator`, `create`, `connect`, `NodeView`
  algebra, rules, `simulate`); `EventProjection` static delivery; `brainevent`
  0.1.0.
- **Imperative / NOT JAX-native (needs rebuild or bypass):**
  - Synapses — all of `_nest/*_synapse.py` (rebuild, Part 2).
  - **Rate/continuous models** — `rate_neuron_ipn/opn`, `lin_rate`, `gauss_rate`,
    `sigmoid_rate*`, `siegert_neuron`, `aeif_cond_alpha_astro`,
    `diffusion_connection` carry the same `_queue` pattern. These are **bucket 3**
    and co-designed with `ContinuousCoupledProj`.
  - Recording — `multimeter` is imperative; the Simulator **bypasses** it by
    tapping `State` into the `for_loop` output (analog recording, built in `02`).
    Do not gut multimeter; tap State.
- **Already done (not a cluster):** §3.1 flagship Brunel family —
  `examples/nest/brunel_{alpha,delta,exp_multisynapse,siegert,alpha_evolution_strategies}.py`
  exist and pass live-NEST parity. Reuse their patterns.
- **NEST upstream sources (read before porting):**
  `/mnt/d/codes/githubs/computational_neuroscience/nest-simulator/pynest/examples/`
  and `.../models/`.

## Part 4 — Conventions (every prompt file)

- **Session protocol** (canonical text in the master index; each prompt embeds a
  self-contained copy):
  1. Create worktree branch `nest-goal/<NN>-<slug>` from `origin/main`
     (`superpowers:using-git-worktrees`).
  2. **Read `CONTEXT.md` fully**, then the named gap-doc sections + relevant source.
  3. Do the work: `superpowers:brainstorming`/`writing-plans` → spec+plan under
     `dev/superpowers/` (gitignored) → TDD (`superpowers:test-driven-development`).
  4. **Live-NEST parity** test in `brainpy_state/_nest/_validation/<x>_test.py`,
     `skipUnless(nest)`. Carve-outs: distributional (not per-sample) where PRNG
     diverges; docs-only items exempt; blocked items ship a skipped placeholder.
  5. NumPy-style docstrings (doctest `Examples` in `.. code-block:: python`);
     tests > 90% coverage on touched code.
  6. **Append a Lessons entry to `CONTEXT.md`** (Part 5 template).
  7. Git lifecycle: commit (**no `Co-Authored-By`**), push, open PR, merge,
     delete remote branch, `git switch main && git merge --ff-only origin/main`,
     remove the worktree.
- **Working agreement (`CLAUDE.md`):** approach-first then wait for approval;
  ask when ambiguous; reflect on every correction; specs/plans gitignored under
  `dev/superpowers/`.

## Part 5 — Lessons learned (APPEND-ONLY; newest at top)

> Each session appends one entry before finishing. Template:
>
> ```
> ### <NN>-<slug> — <YYYY-MM-DD>
> - **Shipped:** <what landed; files; PR #>.
> - **Parity:** <metric: brainpy vs NEST, rel/abs error>.
> - **API discovered/changed:** <new _network primitive shapes, signatures the
>   next cluster should reuse>.
> - **Gotchas:** <NEST-fidelity traps, numerical pitfalls, dt/delay conventions>.
> - **For next clusters:** <advice, blockers found, scope adjustments>.
> ```

_(no entries yet — the first `/goal` session writes here)_
