# /goal — 10 · STDP divergence documentation (trace-storage + `stdp_nn_*` pairing conventions)

> Self-contained prompt. Open at repo root. Stream: synapse (docs carve-out). Depends on: 04, 05 (merged); enriched by 07, 08. Independent of 09. **Docs-only carve-out: doctest, no live-NEST parity run.** This is gap-doc §7 **P0 "Document the STDP trace-storage divergence"** + **P1 "Document spike-pairing-convention parity per `stdp_nn_*`"** — the one deliverable a porting user reads instead of the source.

## Session start

1. Create an isolated worktree branch **`nest-goal/10-stdp-docs`** from `origin/main` (`superpowers:using-git-worktrees`).
2. **Read `CONTEXT.md` (repo root) in full** — the divergences you are documenting are already discovered and frozen in the Lessons; this cluster **consolidates them into user-facing docs**, it does not re-derive them. Pull the exact facts from:
   - **`04-stdp-core` Lessons** — `tau_minus` is a NEST **post-neuron** param (`get_K_value`); the self-contained spec moves it onto the **synapse**; parity sets both. The canonical trace-storage divergence.
   - **`05-stdp-nearest-neighbour` Lessons** — the precise **per-scheme pairing mapping** (`nearest` trace mode; `symm`/`restr`/`pre_centered` eligibility/reset/readout differences; the **phantom-pre-at-0** divergence; facetshw quantisation + key/scope notes). The pairing-convention source of truth.
   - **`07`/`08` Lessons** — param-location moves for `clopath` (`A_LTD/A_LTP/theta_*` neuron→synapse; `delay_u_bars` divergence) and `dopamine` (`tau_minus` post; `n`/`tau_n` on the VT; the online-vs-deferred band) — document these in the same table so the family is covered.
   Then read:
   - `docs/nest-status/internal/synapses-plasticity-gap.md` §5 (trace-storage risks) and §7 P0/P1 (acceptance: a side-by-side `tau_minus` doc with example, **linked from every `stdp_*` docstring**; each NN docstring cites the NEST source line/paper equation).
   - The in-source divergence note already present at `brainpy_state/_nest/stdp_synapse.py` (~lines 51-54) and the `stdp_nn_*` docstrings (05 cited the NEST sources there) — the docs you write **link back to / lift from** these.
   - The docs layout: `ls docs/` and any existing `docs/api/*.rst` / `docs/nest-guide/` / Sphinx `conf.py` + `index.rst` toctree — match the house structure (the gap-doc P0 suggests `docs/nest-guide/` or `docs/api/nest-plasticity.rst`).
3. Write spec + plan under ``. **Approach-first; wait for approval before code.** (Decide the doc file's home + toctree wiring with the user.)

## Goal

One authored, doctested reference page documenting **where STDP state lives in NEST vs brainpy.state** and **each `stdp_nn_*` pairing convention**, cross-linked from every `stdp_*` docstring — so a NEST user porting code finds the divergence without reading source.

## Scope

### A. Trace-storage divergence page (P0)

A new page (e.g. `docs/api/nest-plasticity.rst`, or `docs/nest-guide/stdp-divergences.rst` — confirm in approach-first) with **side-by-side** tables:
- **`tau_minus` / post-trace location:** NEST stores the post-side trace on the **neuron** (`tau_minus` an `archiving_node` param, read via `get_K_value`); brainpy.state's self-contained spec moves it onto the **synapse** (and the substrate stores per-neuron traces). Show the **exact code both ways** (NEST: `nest.SetStatus(neuron, {'tau_minus': …})`; brainpy.state: the spec attr), runnable as a doctest where it doesn't need NEST.
- A **param-location table** spanning the family: for each of `tau_minus` (core/nn/dopamine), `A_LTD/A_LTP/theta_*` (clopath), `n/tau_n` (dopamine VT), `delay_u_bars` (clopath) — *NEST home* vs *brainpy.state home* + the one-line reason (self-containment).
- The **documented numerical divergences** as a short table: clopath `delay_u_bars` (online one-step read; default 4 ms not reproducible) + 5 % band; dopamine online-vs-deferred 0.2 % band; the NN **phantom-pre-at-0**. Each row: what diverges, why, the asserted tolerance, link to the parity test.

### B. `stdp_nn_*` pairing-convention docs (P1)

For each of `stdp_nn_symm_synapse`, `stdp_nn_restr_synapse`, `stdp_nn_pre_centered_synapse` (and a note for `stdp_facetshw_synapse_hom`): a short prose + equation block stating the **exact nearest-neighbour pairing scheme** (which spike pairs with which), citing the **NEST source line / paper equation** (lift the citations 05 already put in the docstrings), with the single-pair regression test named as the executable proof.

### C. Cross-linking

Add a one-line "Parity note" + `:doc:`/`:ref:` link from **every `stdp_*_synapse` (and `clopath_`/`stdp_dopamine_`) docstring** to the relevant section (gap-doc P0 acceptance: *"Linked from every `stdp_*` docstring"*). Keep the existing NumPy-docstring structure; the link goes in **Notes** or **See Also**.

## Constraints

- **Docs carve-out:** doctest only, **no live-NEST parity run** in this cluster (the parity is already proven in 04/05/07/08; here you *cite* it). Examples that need NEST are shown as non-executed literal blocks; examples that run pure brainpy.state are `>>>` doctested.
- **Single source of truth:** pull every number/fact from the CONTEXT.md Lessons and the existing docstrings — do **not** re-measure or restate a divergence differently than the Lessons (if a doc and a Lesson disagree, the Lesson wins; fix the doc).
- Match the repo's Sphinx/RST house style; wire the new page into the toctree so it builds. NumPy-docstring rules (CLAUDE.md) for any touched docstrings.
- Mind the **`__module__='brainpy.state'` doctest-finder** gotcha (04/06/07/08): if you add `>>>` examples to a spec class, they won't be collected by `--doctest-modules`; put runnable doctests in the **`.rst`** (collected by `--doctest-glob='*.rst'`) instead, and verify they pass.

## Validation

- The Sphinx page **builds** (no warnings on the new file) and its doctests pass (`pytest --doctest-glob='*.rst'` on the new page, or the repo's docs-test command — discover it).
- A test asserting **every `stdp_*`/`clopath_`/`stdp_dopamine_` docstring contains the parity-note link** (a small introspection test over the specs — keeps the cross-link from rotting).
- No `requires_nest` test in this cluster (carve-out); if the repo's CI greps for a parity test per touched model, add a docs-exempt marker and note it.

## Edge cases to enumerate + test

A spec whose docstring already has a Notes section (append, don't clobber); a spec class vs an instance docstring; the link target existing for every family member (no dangling `:ref:`); the doctest blocks that **don't** need NEST actually running; RST builds clean under the repo's Sphinx version.

## Definition of done

- [ ] Trace-storage / param-location / numerical-divergence reference page authored, in the toctree, building clean.
- [ ] `stdp_nn_*` pairing conventions documented with NEST source citations.
- [ ] Parity-note link added to every `stdp_*`/`clopath_`/`stdp_dopamine_` docstring; introspection test guards it.
- [ ] Doctests pass; no live-NEST run (carve-out documented); Lessons entry in `CONTEXT.md`; git lifecycle done.

## Session end

1. **Append a Lessons entry to `CONTEXT.md`**: where the doc page lives + how it's wired into Sphinx, the docstring-link introspection-test pattern (reused by future model docs), the doctest-in-`.rst` workaround for the `__module__` filter, and the now-complete map of documented STDP divergences. Note this **closes the discrete-synapse stream's docs/validation** — the next synapse work is the bucket-3 continuous **re-grill**.
2. Verify the docs build + doctests pass; docstrings done.
3. Commit (**no `Co-Authored-By`**) → push → PR (scope + the divergence map) → merge → delete remote branch → `git switch main && git merge --ff-only origin/main` → remove worktree.
