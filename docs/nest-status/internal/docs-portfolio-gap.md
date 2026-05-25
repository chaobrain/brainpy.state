# Docs portfolio — NEST parity gap

## 1. Scope

NEST's user-doc tiers (getting-started, networks, synapses/connectivity,
neuron models, devices, recording, stimulating, randomness, parallel,
tutorials, PyNEST API reference, glossary) vs. the repo's `docs/` tree.
Identify which tiers exist, which are partial, and which are absent. The
output is a list of doc-tier porting targets.

Upstream reference: <https://nest-simulator.readthedocs.io/en/stable/>

Evidence basis:
- Repo doc inventory (run 2026-05-11):
  ```
  docs/index.rst
  docs/changelog.md
  docs/api/{base,brainpy-*,nest-*,index}.rst
  docs/brainpy-guide/{architecture,neurons,synapses,projections}.ipynb + index.rst
  docs/quickstart/{5min-tutorial,overview}.ipynb + installation.rst + index.rst
  docs/nest-status/index.rst (the public Experimental caveats page)
  docs/examples/gallery.rst
  ```
- Repo NEST API reference (`docs/api/nest-neurons.rst`, lines 1-210, read in
  this analysis): clean autosummary stubs organized into 12 categories (IAF
  psc, IAF cond, IAF specialized, AdEx, GIF, MAT, GLIF, HH, Izhikevich, point
  process, binary, rate, other). Header carries the Experimental warning and
  cross-references `nest-status/index`. No parameter tables, no NEST upstream
  links, no narrative usage examples in the stub file itself (although the
  generated per-class doc pages may render the docstring tables; this needs
  building to confirm).

## 2. Parity summary

Repo has the foundational tiers (quickstart, brainpy-guide, API reference,
public NEST caveats). The biggest gap is the **absence of a `docs/nest-guide/`
porting tutorial tier** — the doc-portfolio analog of `network-api-gap.md`'s
P0 (`nest_compat` shim). Several other NEST tiers (Connection management,
Recording from simulations, Randomness, Tutorials) are absent. The
Parallel-computing tier is intentionally divergent (JAX device sharding vs.
MPI).

| Bucket | Count | Notes |
|---|---:|---|
| implemented | 1 | quickstart + brainpy-guide cover brainpy-style modeling well |
| partial | 4 | API ref (autosummary stubs only); examples (gallery without ports); installation (no NEST-compat install guidance); changelog (present but not NEST-segmented) |
| missing | 6 | `docs/nest-guide/` porting tutorial; Connection management guide; Recording from simulations guide; Randomness in JAX vs. NEST guide; PyNEST API mapping reference; NEST tutorials series |
| divergent | 1 | Parallel computing (JAX device sharding instead of MPI — needs a dedicated doc) |
| unsupported | 1 | NESTML/SLI modeling-language documentation (spec §7) |

## 3. Evidence-backed mapping table

| NEST doc tier | Status | brainpy.state location | NEST upstream | Notes |
|---|---|---|---|---|
| Installation | partial | `docs/quickstart/installation.rst` | <https://nest-simulator.readthedocs.io/en/stable/installation/index.html> | exists, but no NEST-compat / extras-installation guidance; NEST users expect MPI/Python-version notes |
| Getting started | implemented | `docs/quickstart/5min-tutorial.ipynb`, `overview.ipynb` | <https://nest-simulator.readthedocs.io/en/stable/get_started/get_started.html> | brainpy-style; doesn't cover NEST-compat surface (which is absent — `network-api-gap.md`) |
| **PyNEST porting guide** | **missing** | none | n/a (this is the user-facing analog of `network-api-gap.md` P0) | **most consequential gap** — see roadmap P0 |
| Tutorials (multi-part PyNEST series) | missing | `docs/brainpy-guide/` covers brainpy style only | <https://nest-simulator.readthedocs.io/en/stable/tutorials/pynest_tutorial/index.html> | NEST's 4-part PyNEST tutorial walks new users through Create → Connect → Simulate → Plot |
| Networks (Connection management) | missing | none | <https://nest-simulator.readthedocs.io/en/stable/synapses/connectivity_concepts.html> | NEST's flagship Connect doc; absent from repo |
| Spatially structured networks | missing | none | <https://nest-simulator.readthedocs.io/en/stable/networks/spatially_structured_networks.html> | blocked by `network-api-gap.md` spatial roadmap |
| Synapses / plasticity guide | partial | `docs/api/nest-synapses.rst`, `nest-plasticity.rst` (autosummary stubs) | <https://nest-simulator.readthedocs.io/en/stable/synapses/index.html> | API references only; no narrative around spike-pairing, plasticity selection, weight clipping |
| Neuron models reference | partial | `docs/api/nest-neurons.rst` (lines 1-210 — clean stubs, 12 categories) | <https://nest-simulator.readthedocs.io/en/stable/models/index_neuron.html> | parameter tables and NEST cross-references missing from the stub file (docstrings themselves are detailed, e.g., `iaf_psc_alpha.py:193-274` has a parameter-mapping table; consider auto-pulling these in) |
| Devices reference | partial | `docs/api/nest-devices.rst` | <https://nest-simulator.readthedocs.io/en/stable/models/index_device.html> | stub only |
| Recording from simulations guide | missing | none | <https://nest-simulator.readthedocs.io/en/stable/connect_nest/connections_with_simulator.html> (and recording-specific pages) | no narrative around multimeter / spike_recorder / weight_recorder usage; cross-link `devices-gap.md` |
| Stimulating networks guide | missing | none | upstream | NEST publishes a tour of stimulation devices |
| Randomness in NEST | missing | none | <https://nest-simulator.readthedocs.io/en/stable/nest_behavior/random_numbers.html> | a JAX-PRNG-vs-NEST-PRNG-discussion doc would be load-bearing for any user reproducing NEST seeds; spec §7 establishes that bit-exact reproduction is unsupported |
| Parallel computing | divergent (no doc) | none | <https://nest-simulator.readthedocs.io/en/stable/hpc/parallel_computing.html> | brainpy.state uses JAX device sharding; a "Parallel computing in brainpy.state vs. NEST" doc would lay out the mapping |
| PyNEST API reference | missing | none | <https://nest-simulator.readthedocs.io/en/stable/ref_material/pynest_api/index.html> | blocked by absence of the API surface (`network-api-gap.md`); even a "see the corresponding brainpy.state idiom" mapping table would be load-bearing |
| Glossary | missing | none | <https://nest-simulator.readthedocs.io/en/stable/ref_material/glossary.html> | useful for NEST users learning brainpy.state terminology |
| Examples gallery | partial | `docs/examples/gallery.rst` | <https://nest-simulator.readthedocs.io/en/stable/examples/index.html> | brainpy-style only; cross-link `examples-gap.md` |
| Public NEST-status page (caveats) | implemented | `docs/nest-status/index.rst` | n/a | the authoritative user-facing scope/limitations page |
| Internal gap analysis | implemented (this document set) | `docs/nest-status/internal/` | n/a | excluded from Sphinx build |
| NESTML / SLI modeling-language docs | unsupported | n/a | upstream | spec §7 — brainpy.state authors models in Python directly |
| Changelog | partial | `docs/changelog.md` | n/a | exists; doesn't segment NEST vs. brainpy-style changes |

## 4. Missing or incomplete functionality

**Tier-level gaps (6):**

- **`docs/nest-guide/`** — porting tutorial from PyNEST → brainpy.state. Single
  most consequential doc gap. Should cover: (a) creating neurons, (b) creating
  generators, (c) connecting populations, (d) recording, (e) simulation +
  Run/Cleanup, (f) plasticity, (g) plotting results. Each step side-by-side
  as a PyNEST snippet + brainpy.state equivalent (using `nest_compat` once it
  exists, raw API meanwhile).
- **Connection management guide** — explain what `pairwise_bernoulli`,
  `fixed_indegree`, etc. mean in NEST and how to express each in brainpy.state.
  Blocked by absence of the API surface but the *concepts* can be documented
  now.
- **Recording from simulations guide** — narrative around `multimeter`,
  `spike_recorder`, `weight_recorder` usage, gating semantics, and
  recording-backend divergence (cross-link `devices-gap.md`).
- **Randomness guide** — explain that brainpy.state uses JAX PRNG keys, NEST
  uses per-thread RNGs, that bit-exact parity is impossible (spec §7), and
  what *distributional* parity means in practice. Reference the harness
  conventions from `numerical-validation-gap.md`.
- **PyNEST API mapping reference** — a one-page table that maps every PyNEST
  top-level function to its brainpy.state idiom (or "missing — use
  `nest_compat`"). Effectively a user-facing version of
  `network-api-gap.md` §3.
- **NEST tutorials series** — `docs/nest-guide/tutorials/` covering equivalents
  of NEST's 4-part PyNEST tutorial. Cross-link `examples-gap.md` P0 (port
  `one_neuron.py`, etc.).

**Per-tier improvements (4):**

- **Neuron-models reference (`docs/api/nest-neurons.rst`)**: add a parameter
  table per family + an upstream link per model. Parameter tables already
  live in individual source-file docstrings (e.g.
  `brainpy_state/_nest/iaf_psc_alpha.py:193-274`), so the work is to make
  them render in autosummary or add them to the per-family rst.
- **Devices reference (`docs/api/nest-devices.rst`)**: same as above plus a
  pointer to `devices-gap.md` recording-semantics caveats.
- **Examples gallery (`docs/examples/gallery.rst`)**: add a "NEST porting
  examples" section (initially empty, filled by `examples-gap.md` ports as
  they land).
- **Changelog**: segment NEST-compat changes into their own subsection so
  NEST users tracking parity progress can find changes affecting them.

## 5. Semantic & numerical risks

This section is mostly downstream of other docs — risks here are about
**user-comprehension mismatch**, not numerical drift.

- **`tau_minus` storage** (cross-link `synapses-plasticity-gap.md` §5): users
  porting NEST STDP code will set `tau_minus` on the wrong object and get
  silently different behavior. Without a porting guide, this happens.
- **`SetStatus(neuron, {'V_m': v})` pattern**: NEST users expect dict-style
  status mutation; brainpy.state uses attribute mutation. Without a porting
  guide cheatsheet, every NEST user hits this in the first 5 minutes.
- **`Connect` programming model**: NEST users expect `Connect` to mutate
  global state. brainpy.state requires composition. Without a porting guide,
  this is the first compilation failure they hit.
- **Brainpy-extension parameter convention**: every NEST neuron in repo
  carries `V_initializer`, `spk_fun`, `spk_reset`, `ref_var` in addition to
  NEST parameters (`neurons-gap.md` §5). Without a doc, NEST users
  question whether these are required or optional.
- **Recording-device semantic divergence**: `nest-status/index.rst:92-93`
  flags this but in vague language. A concrete doc + reproducer (cross-link
  `devices-gap.md` P0) sharpens the user picture.
- **Randomness expectations**: NEST users habitually use `SetKernelStatus({'rng_seed': N})`
  and expect bit-exact reproducibility; in JAX they'd thread `PRNGKey`s.
  Without a doc explaining the mental shift, expectations diverge silently.

## 6. Validation gaps

These don't apply directly — docs aren't validated numerically — but two
quality-gate checks should be in place:

- **Sphinx build cleanliness**: confirm `nest-status/internal/` is excluded
  (done in Task 0) and that no public page links into it.
- **Cross-reference integrity**: every cross-link from a per-axis gap doc to
  another (e.g. `network-api-gap.md` → `docs-portfolio-gap.md`) should
  resolve. To be checked in Task 10.
- **Code example correctness**: when `docs/nest-guide/` lands, every PyNEST
  snippet in it should run against NEST and every brainpy.state snippet
  should run against the repo. A simple `doctest` or notebook-CI step.

## 7. Prioritized roadmap

- **P0 — Create `docs/nest-guide/` and write the porting tutorial.** [L]
  Rationale: most consequential single doc improvement. Acceptance:
  `docs/nest-guide/index.rst` exists and the tutorial covers Create →
  Connect → Simulate → Plot side-by-side as PyNEST + brainpy.state (using
  `nest_compat` once available, or raw brainpy.state idioms meanwhile);
  linked from the public `nest-status/index.rst` Experimental warning;
  buildable in Sphinx.

- **P0 — Add a PyNEST→brainpy.state cheatsheet.** [S]
  Part of `docs/nest-guide/` (subpage). Acceptance: a single table that maps
  `Create`, `Connect`, `Simulate`, `GetStatus`, `SetStatus`, `CopyModel`,
  `ResetKernel`, `SetKernelStatus`, plus the 4 most-used connection rules,
  to the corresponding brainpy.state idiom. Lives at
  `docs/nest-guide/cheatsheet.rst`.

- **P0 — Parameter-table render in `docs/api/nest-neurons.rst` and
  `nest-synapses.rst`.** [M]
  Rationale: parameter tables exist in docstrings but Sphinx autosummary
  doesn't display them in the family-overview page; users have to click
  through to each model. Acceptance: each model name in `nest-neurons.rst`
  has a "Defaults" mini-table immediately under it (parameter, default,
  unit, NEST upstream link).

- **P1 — Recording-from-simulations guide.** [M]
  Cross-link `devices-gap.md`. Acceptance: `docs/nest-guide/recording.rst`
  covers multimeter usage, spike_recorder gating semantics, weight_recorder
  hookup per plasticity rule, and the recording-backend gap.

- **P1 — Connection management guide.** [M]
  Cross-link `network-api-gap.md`. Acceptance: `docs/nest-guide/connections.rst`
  documents the 10 NEST connection rules with worked brainpy.state examples;
  starts with `nest_compat` calls then shows the underlying brainstate
  primitive each rule maps to.

- **P1 — Randomness guide.** [S]
  Acceptance: `docs/nest-guide/randomness.rst` explains JAX PRNG keys vs.
  NEST per-thread RNGs, states the distributional-vs-bitwise parity
  decision, and shows a seed-passing example that gets reproducible results
  across runs in brainpy.state.

- **P1 — Tutorials series (PyNEST 4-part equivalents).** [L]
  Cross-link `examples-gap.md` P0. Acceptance: 4 tutorial notebooks under
  `docs/nest-guide/tutorials/` covering the 4 parts of NEST's PyNEST
  tutorial; each notebook runnable end-to-end.

- **P1 — PyNEST API mapping reference.** [M]
  User-facing version of `network-api-gap.md` §3. Acceptance:
  `docs/nest-guide/pynest-api-map.rst` lists every PyNEST function with
  its brainpy.state idiom (or "missing — use `nest_compat`"); auto-generated
  from a stable data file if possible (matches the catalog snapshot pattern).

- **P2 — NEST-style parameter-table convention for `docs/api/nest-devices.rst`
  and `nest-plasticity.rst`.** [S]
  Acceptance: same as the P0 parameter-table render, extended to devices
  and plasticity stub files.

- **P2 — Parallel-computing-in-brainpy.state guide.** [M]
  Rationale: NEST users porting MPI scripts to brainpy.state need to know
  about `jax.pmap`, device sharding, and per-device PRNG. Acceptance:
  `docs/nest-guide/parallel.rst` covers the mapping with a runnable
  small-network parallel example.

- **P2 — Glossary.** [S]
  Acceptance: `docs/nest-guide/glossary.rst` defines NEST terms (NodeCollection,
  SynapseCollection, kernel, virtual process, slice, ring buffer, archiving
  node) and points to brainpy.state equivalents (or "no equivalent — see X").

- **P2 — Segment changelog by NEST vs. brainpy-style changes.** [S]
  Acceptance: `docs/changelog.md` gains a top-level "NEST-compat" subsection;
  changes affecting NEST parity are entered there.

- **P2 — Doc-CI: doctest/notebook-CI for the porting tutorial.** [M]
  Acceptance: a GitHub Actions job runs the tutorial code blocks against
  a NEST-installed runner and the repo to catch drift between docs and
  code as parity work proceeds.
