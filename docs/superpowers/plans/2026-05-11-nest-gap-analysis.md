# NEST Gap Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an internal, maintainer-only NEST parity gap analysis (1 index + 7 per-axis docs + catalog snapshot) under `docs/nest-status/internal/`, following the methodology, classification taxonomy, and evidence rules in the approved spec at `docs/superpowers/specs/2026-05-11-nest-gap-analysis-design.md`.

**Architecture:** Structural-sweep methodology — one catalog-snapshot task seeds the upstream feature set, then seven independent per-axis doc tasks each diff `brainpy_state/_nest/` against the relevant slice of the catalog and emit a uniformly-templated gap doc. A final index task rolls everything up. Per-axis docs are independent and parallelizable after the catalog snapshot lands.

**Tech Stack:** Markdown documents (Sphinx-excluded), `WebFetch` for upstream NEST docs, `grep`/`Read` over `brainpy_state/_nest/`, git for atomic commits.

**Spec reference (authoritative):** `docs/superpowers/specs/2026-05-11-nest-gap-analysis-design.md` — every classification value, evidence rule, and acceptance criterion comes from there.

**Working notes:**
- This is documentation/analysis work, not code work. TDD doesn't apply. The verification step in each task is a checklist against spec §4 (evidence rules) and §8 (acceptance criteria) instead of a passing test.
- Each per-axis doc uses the **identical template** defined in spec §2.3. Copy it verbatim per task.
- The only allowed status values in mapping tables: `implemented`, `unvalidated`, `partial`, `divergent`, `missing`, `unsupported`. Spec §3.
- NEST reference version: 3.x latest stable on `nest-simulator.readthedocs.io`. Record exact retrieval date in the catalog snapshot.
- Commit at the end of each task. Conventional commits style (`docs(nest-gap): ...`).

---

## Task 0: Sphinx exclusion + directory scaffold

**Files:**
- Modify: `docs/conf.py` (add `exclude_patterns` entry for `nest-status/internal/**`)
- Create: `docs/nest-status/internal/.gitkeep`

- [ ] **Step 1: Inspect current `docs/conf.py` to find the right insertion point.**

Run: `grep -n "exclude_patterns\|html_extra_path\|source_suffix" docs/conf.py`

Expected: there should be an `exclude_patterns` list. If absent, add a new `exclude_patterns = [...]` line near other top-level Sphinx config.

- [ ] **Step 2: Add `'nest-status/internal/**'` to `exclude_patterns`.**

If `exclude_patterns` already exists, extend it. Example final form:

```python
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'nest-status/internal/**']
```

If it doesn't exist, add it as a new top-level line in `docs/conf.py`.

- [ ] **Step 3: Create the internal directory with a `.gitkeep`.**

```bash
mkdir -p docs/nest-status/internal
touch docs/nest-status/internal/.gitkeep
```

- [ ] **Step 4: Verify Sphinx build still works and the dir is excluded.**

Run from `docs/`: `sphinx-build -b html -W --keep-going . _build/html 2>&1 | tail -20`

Expected: build completes (warnings tolerated initially; the `-W` flag promotes warnings to errors so any new warning surfaces). No file under `_build/html/nest-status/internal/` exists.

If `sphinx-build` is not available, skip this step and note in the commit message that the exclusion was added but not locally verified.

- [ ] **Step 5: Commit.**

```bash
git add docs/conf.py docs/nest-status/internal/.gitkeep
git commit -m "docs(nest-gap): scaffold internal gap-analysis dir and exclude from Sphinx"
```

---

## Task 1: NEST catalog snapshot

**Files:**
- Create: `docs/nest-status/internal/nest-catalog-snapshot.md`

**Purpose:** Freeze the upstream NEST 3.x feature set into a single checked-in file so every per-axis doc diffs against the same baseline and future re-diffs are cheap.

- [ ] **Step 1: Fetch the NEST models index.**

```
WebFetch url=https://nest-simulator.readthedocs.io/en/stable/models/index.html
        prompt="List every model on this page grouped by category (Neuron, Synapse, Device, Generator, Recorder, Detector, Rate, Binary, Multi-compartment, Astrocyte). For each model, include the model name, one-line description, and the relative URL to its individual doc page."
```

If the page is paginated or the prompt result is truncated, fetch each category subpage individually. The likely URLs are:
- `https://nest-simulator.readthedocs.io/en/stable/models/index_neuron.html`
- `https://nest-simulator.readthedocs.io/en/stable/models/index_synapse.html`
- `https://nest-simulator.readthedocs.io/en/stable/models/index_device.html` (or `index_stimulation_device.html`/`index_recording_device.html`)
- `https://nest-simulator.readthedocs.io/en/stable/models/index_rate.html`
- `https://nest-simulator.readthedocs.io/en/stable/models/index_binary.html`

- [ ] **Step 2: Fetch the connection management API.**

```
WebFetch url=https://nest-simulator.readthedocs.io/en/stable/networks/connection_management.html
        prompt="List every connection rule (one_to_one, all_to_all, fixed_indegree, fixed_outdegree, fixed_total_number, pairwise_bernoulli, pairwise_poisson, symmetric variants, etc.) with a one-line description and the parameter dict shape NEST expects."
```

- [ ] **Step 3: Fetch the PyNEST top-level API reference.**

```
WebFetch url=https://nest-simulator.readthedocs.io/en/stable/ref_material/pynest_apis.html
        prompt="List every top-level PyNEST function (Connect, Create, CopyModel, GetStatus, SetStatus, GetConnections, NodeCollection constructor, Simulate, ResetKernel, SetKernelStatus, GetKernelStatus, spatial/topology API, Parameter expressions). For each, include the signature and one-line semantic summary."
```

- [ ] **Step 4: Fetch the spatial/topology API.**

```
WebFetch url=https://nest-simulator.readthedocs.io/en/stable/networks/spatially_structured_networks.html
        prompt="List the spatial/topology entities: Layer, Mask, Parameter, plus the placement and connection rules specific to spatial networks. One line per entity."
```

- [ ] **Step 5: Compose `nest-catalog-snapshot.md`.**

Template:

```markdown
# NEST 3.x Catalog Snapshot

**Retrieved:** YYYY-MM-DD
**NEST version target:** 3.x (latest stable per nest-simulator.readthedocs.io/en/stable/)
**Purpose:** Frozen baseline for the gap analysis in this directory. Re-run the fetch
steps in `../../superpowers/plans/2026-05-11-nest-gap-analysis.md` Task 1 to refresh.

## Neurons

| Model | Category | Description | Upstream doc |
|---|---|---|---|
| ... | ... | ... | ... |

## Synapses and plasticity

| Model | Class | Description | Upstream doc |
|---|---|---|---|
| ... | ... | ... | ... |

## Devices — generators / stimulation

| Model | Description | Upstream doc |
|---|---|---|

## Devices — recorders / detectors

| Model | Description | Upstream doc |
|---|---|---|

## Rate models

| Model | Description | Upstream doc |
|---|---|---|

## Binary models

| Model | Description | Upstream doc |
|---|---|---|

## Multi-compartment / structured neurons

| Model | Description | Upstream doc |
|---|---|---|

## Astrocyte / glial models

| Model | Description | Upstream doc |
|---|---|---|

## Connection rules (Connect())

| Rule | Required params | Description | Upstream doc |
|---|---|---|---|

## PyNEST top-level API surface

| Function | Signature summary | Purpose | Upstream doc |
|---|---|---|---|

## Spatial / topology API

| Entity | Purpose | Upstream doc |
|---|---|---|

## Kernel / simulator surface

| Function | Purpose | Upstream doc |
|---|---|---|
| ResetKernel | Reset simulator state | ... |
| SetKernelStatus / GetKernelStatus | Configure kernel | ... |
| Simulate | Run for given biological time | ... |
| Prepare / Run / Cleanup | Split simulation lifecycle | ... |
```

Fill every row with data from the WebFetch results. Do not leave placeholders.

- [ ] **Step 6: Verify the snapshot is complete.**

Run: `wc -l docs/nest-status/internal/nest-catalog-snapshot.md`

Expected: file is substantial (≥ 200 lines) and every section is populated. If any section is empty, repeat Step 1-4 for that section.

- [ ] **Step 7: Commit.**

```bash
git add docs/nest-status/internal/nest-catalog-snapshot.md
git rm docs/nest-status/internal/.gitkeep
git commit -m "docs(nest-gap): freeze NEST 3.x catalog snapshot"
```

---

## Task 2: Neurons gap doc

**Files:**
- Create: `docs/nest-status/internal/neurons-gap.md`
- Read (representative implementations):
  - `brainpy_state/_nest/iaf_psc_alpha.py`
  - `brainpy_state/_nest/iaf_psc_exp_ps.py`
  - `brainpy_state/_nest/iaf_psc_exp_ps_lossless.py`
  - `brainpy_state/_nest/iaf_cond_alpha.py`
  - `brainpy_state/_nest/iaf_cond_alpha_mc.py`
  - `brainpy_state/_nest/aeif_cond_alpha.py`
  - `brainpy_state/_nest/aeif_psc_delta_clopath.py`
  - `brainpy_state/_nest/gif_psc_exp.py`
  - `brainpy_state/_nest/glif_psc.py`
  - `brainpy_state/_nest/hh_psc_alpha.py`
  - `brainpy_state/_nest/hh_cond_beta_gap_traub.py`
  - `brainpy_state/_nest/ht_neuron.py`
  - `brainpy_state/_nest/izhikevich.py`
  - `brainpy_state/_nest/mat2_psc_exp.py`
  - `brainpy_state/_nest/lin_rate.py`
  - `brainpy_state/_nest/siegert_neuron.py`
  - `brainpy_state/_nest/erfc_neuron.py`
  - `brainpy_state/_nest/pp_psc_delta.py`
  - `brainpy_state/_nest/cm_default.py`
  - `brainpy_state/_nest/ignore_and_fire.py`
- Read (catalog reference): `docs/nest-status/internal/nest-catalog-snapshot.md` (neuron + rate + binary + multi-compartment sections)

**Scope:** IAF (psc/cond/multisynapse/ps/lossless), AdEx, GIF, GLIF, HH, MAT, Izhikevich, rate, binary, point-process, multi-compartment, astrocyte, ignore_and_fire. Pull the ported list from `brainpy_state/_nest/__init__.py`.

- [ ] **Step 1: Enumerate ported neurons.**

```bash
grep -E "^from \.[a-z_0-9]+ import" brainpy_state/_nest/__init__.py | \
  sed 's/.*from \.\([a-z_0-9]*\) import.*/\1/' | sort -u > /tmp/ported.txt
wc -l /tmp/ported.txt
```

Expected: ~118 ported module names. Cross-reference each line against the catalog snapshot to bucket as `implemented` candidate, `unvalidated`, `partial`, `divergent`, or `unsupported`.

- [ ] **Step 2: For each lead implementation in the "Read" list, extract parameters/defaults/units/state.**

For each file, use `Read` to capture:
- `__init__` signature: parameter names + defaults + units.
- `init_state`: state-variable keys + initial values + units.
- `update` method: integration approach (RKF45 via `AdaptiveRungeKuttaStep`, analytical propagator, or simple Euler/explicit).
- Any `jit_error_if` calls (boundary checks) and any Python `raise` calls (red flag — should be jit_error_if per spec §5).

For each, also `Read` the paired `*_test.py` and `grep -l "import nest" brainpy_state/_nest/<file>_test.py` to flag whether numerical comparison vs. NEST exists.

- [ ] **Step 3: For each lead implementation, fetch the NEST upstream doc page.**

Example for `iaf_psc_alpha`:

```
WebFetch url=https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_alpha.html
        prompt="List the model's parameters (name, default, unit, description), state variables (name, unit, description), and how spikes are emitted (threshold, reset, refractory). Also note the integration scheme and any documented limitations."
```

Repeat for every lead implementation. Cache the per-model upstream URL in a scratch list — those URLs go into the mapping table.

- [ ] **Step 4: Diff each lead implementation against its upstream doc.**

Record per model: parameter set match (count + any name mismatches), default value drift (any?), unit drift (saiunit vs. NEST's units), state-variable layout match, integration method note, validation flag (test imports `nest`? yes/no).

Use these findings to classify each lead. Extrapolate to siblings in the same family with a one-sentence note (e.g., "iaf_psc_exp and iaf_psc_alpha_multisynapse extrapolated from iaf_psc_alpha; same parameter naming convention observed").

- [ ] **Step 5: Compose `neurons-gap.md` using the spec §2.3 template.**

Template (copy verbatim into the file, then fill):

```markdown
# Neurons — NEST parity gap

## 1. Scope

Covers IAF (psc/cond/multisynapse/ps/lossless), AdEx (aeif_*), GIF, GLIF, HH,
MAT, Izhikevich, rate (lin_rate, siegert_neuron, tanh_rate, sigmoid_rate*,
threshold_lin_rate, gauss_rate), binary (erfc_neuron, ginzburg_neuron,
mcculloch_pitts_neuron), point-process (pp_psc_delta, pp_cond_exp_mc_urbanczik),
multi-compartment (cm_default, iaf_cond_alpha_mc), astrocyte (astrocyte_lr_1994),
and `ignore_and_fire`.

Upstream reference: https://nest-simulator.readthedocs.io/en/stable/models/index_neuron.html

## 2. Parity summary

<one paragraph summarizing the headline finding>

| Bucket | Count |
|---|---|
| implemented | N |
| unvalidated | N |
| partial | N |
| divergent | N |
| missing | N |
| unsupported | N |
| **total NEST neurons surveyed** | N |

## 3. Evidence-backed mapping table

| NEST model | Status | brainpy.state location | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| iaf_psc_alpha | <status> | `brainpy_state/_nest/iaf_psc_alpha.py` | https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_alpha.html | `iaf_psc_alpha_test.py` (nest=Y/N) | <one-line gap or risk> |
| ... | | | | | |

(One row per NEST neuron in the catalog snapshot. Include rows for missing
models with `_nest/` location empty.)

## 4. Missing or incomplete functionality

- `<nest_model_name>` — missing entirely. Upstream doc: <url>. Notes: <why it
  matters, e.g., used in standard benchmarks>.
- `<nest_model_name>` — partial: missing `<parameter_or_feature>` (see
  `brainpy_state/_nest/<file>.py:<line>`).

## 5. Semantic & numerical risks

- **Defaults drift**: `<model>` — `<param>` default is `<repo_value>` vs.
  NEST's `<nest_value>` (see `brainpy_state/_nest/<file>.py:<line>`).
- **Unit divergence**: ...
- **State-variable layout**: ...
- **Integration method**: ...
- **Stochasticity**: PRNG seeding semantics differ (JAX PRNGKey vs. NEST per-
  thread RNG). Bit-exact reproduction impossible; distributional comparison
  feasible.
- **Delay handling**: NEST uses min-delay-based slice scheduling; brainpy.state
  uses brainstate's delay infrastructure (note any models that rely on
  precise delays).
- **Refractory handling**: ...

## 6. Validation gaps

- No `import nest` reference test for: `<list of families>`.
- Existing NEST-comparison tests cover: `<list>`. Document tolerance and
  duration.

## 7. Prioritized roadmap

- **P0 — Build NEST-comparison harness for IAF family.** [M]
  Rationale: <…>. Acceptance: `iaf_psc_alpha`, `iaf_psc_exp`, `iaf_cond_alpha`
  reproduce NEST membrane-potential trace within X% over a 1s window with
  matched seeds and matched dt; harness lives at
  `brainpy_state/_nest/_validation/nest_compare.py`.

- **P1 — <item>** [<size>] — <rationale + acceptance>.

- **P2 — <item>** [<size>] — <rationale + acceptance>.
```

- [ ] **Step 6: Verify §4 evidence rules are met.**

Run: `grep -c "brainpy_state/_nest/" docs/nest-status/internal/neurons-gap.md`

Expected: every mapping-table row has a repo path or a `_missing_` marker. No row has empty `brainpy.state location` for an `implemented`/`unvalidated`/`partial`/`divergent` row.

Run: `grep -c "nest-simulator.readthedocs.io" docs/nest-status/internal/neurons-gap.md`

Expected: every row carries an upstream link. Count ≥ row count of the mapping table.

- [ ] **Step 7: Verify every roadmap item has an acceptance criterion.**

Inspect §7. Each bullet must contain the literal substring "Acceptance:". If not, add one.

- [ ] **Step 8: Commit.**

```bash
git add docs/nest-status/internal/neurons-gap.md
git commit -m "docs(nest-gap): add neurons gap doc"
```

---

## Task 3: Synapses + plasticity gap doc

**Files:**
- Create: `docs/nest-status/internal/synapses-plasticity-gap.md`
- Read (representative implementations):
  - `brainpy_state/_nest/static_synapse.py`
  - `brainpy_state/_nest/static_synapse_hom_w.py`
  - `brainpy_state/_nest/stdp_synapse.py`
  - `brainpy_state/_nest/stdp_dopamine_synapse.py`
  - `brainpy_state/_nest/stdp_triplet_synapse.py`
  - `brainpy_state/_nest/stdp_facetshw_synapse_hom.py`
  - `brainpy_state/_nest/stdp_pl_synapse_hom.py`
  - `brainpy_state/_nest/stdp_nn_pre_centered_synapse.py`
  - `brainpy_state/_nest/stdp_nn_restr_synapse.py`
  - `brainpy_state/_nest/stdp_nn_symm_synapse.py`
  - `brainpy_state/_nest/tsodyks_synapse.py`
  - `brainpy_state/_nest/tsodyks2_synapse.py`
  - `brainpy_state/_nest/quantal_stp_synapse.py`
  - `brainpy_state/_nest/clopath_synapse.py`
  - `brainpy_state/_nest/urbanczik_synapse.py`
  - `brainpy_state/_nest/jonke_synapse.py`
  - `brainpy_state/_nest/vogels_sprekeler_synapse.py`
  - `brainpy_state/_nest/volume_transmitter.py`
  - `brainpy_state/_nest/gap_junction.py`
  - `brainpy_state/_nest/sic_connection.py`
  - `brainpy_state/_nest/diffusion_connection.py`
  - `brainpy_state/_nest/cont_delay_synapse.py`
  - `brainpy_state/_nest/bernoulli_synapse.py`
  - `brainpy_state/_nest/ht_synapse.py`
- Read (catalog reference): `docs/nest-status/internal/nest-catalog-snapshot.md` (synapse section)

**Scope:** Static synapses, STDP (9 variants), STP (tsodyks/quantal), Clopath, Urbanczik, Jonke, Vogels-Sprekeler, dopamine, volume_transmitter, gap junctions, SIC, diffusion, continuous-delay, bernoulli, ht_synapse.

- [ ] **Step 1: Enumerate ported synapses/plasticity rules.**

```bash
grep -E "^from \.[a-z_0-9]+(synapse|junction|connection|transmitter|stdp|tsodyks|stp|clopath|urbanczik) import" brainpy_state/_nest/__init__.py | \
  sed 's/.*from \.\([a-z_0-9]*\) import.*/\1/' | sort -u
```

Cross-reference each against the catalog snapshot.

- [ ] **Step 2: For each lead implementation, extract weight-update rule, pre/post traces, state vars, and parameter defaults.**

Per file, `Read` and capture:
- Init signature: weight, delay, learning-rate parameters with units.
- State variables: pre-trace, post-trace, weight bounds.
- Update rule: where the synaptic update happens, what triggers it.
- Whether `volume_transmitter`-style modulation is wired in.
- Whether `weight_recorder` hooks exist.
- Paired test file: `grep -l "import nest" brainpy_state/_nest/<name>_test.py`.

- [ ] **Step 3: Fetch upstream docs for each lead rule.**

Example:

```
WebFetch url=https://nest-simulator.readthedocs.io/en/stable/models/stdp_synapse.html
        prompt="List the synapse parameters with defaults and units, the weight-update equations, the pre/post-trace dynamics, and any documented limitations. Note whether weight_recorder is supported."
```

Repeat for every lead.

- [ ] **Step 4: Diff and classify.**

Per rule: parameter match, default drift, weight-update equation match, trace dynamics match, weight_recorder hookup, validation flag.

- [ ] **Step 5: Compose `synapses-plasticity-gap.md` using the spec §2.3 template.**

Use the same template structure as Task 2 (header, scope, parity summary, mapping table, missing/incomplete, semantic risks, validation gaps, roadmap). Substitute "synapses and plasticity rules" for "neurons" throughout. Make the mapping table include one row per upstream NEST synapse model.

Semantic-risk section must explicitly call out:
- **Spike-pairing convention**: nearest-neighbor vs. all-to-all pairing — does each repo STDP variant match NEST's documented pairing?
- **Weight clipping**: bounds enforcement — NEST clamps in the update; what does the repo do?
- **Volume transmitter latency**: dopamine delay handling.
- **Weight recorder integration**: whether plasticity rules emit weight-recorder events.

- [ ] **Step 6: Verify §4 evidence rules.**

Same checks as Task 2 Step 6 + 7.

- [ ] **Step 7: Commit.**

```bash
git add docs/nest-status/internal/synapses-plasticity-gap.md
git commit -m "docs(nest-gap): add synapses+plasticity gap doc"
```

---

## Task 4: Devices gap doc

**Files:**
- Create: `docs/nest-status/internal/devices-gap.md`
- Read (representative implementations):
  - `brainpy_state/_nest/ac_generator.py`
  - `brainpy_state/_nest/dc_generator.py`
  - `brainpy_state/_nest/step_current_generator.py`
  - `brainpy_state/_nest/step_rate_generator.py`
  - `brainpy_state/_nest/poisson_generator.py`
  - `brainpy_state/_nest/poisson_generator_ps.py`
  - `brainpy_state/_nest/inhomogeneous_poisson_generator.py`
  - `brainpy_state/_nest/sinusoidal_poisson_generator.py`
  - `brainpy_state/_nest/sinusoidal_gamma_generator.py`
  - `brainpy_state/_nest/gamma_sup_generator.py`
  - `brainpy_state/_nest/ppd_sup_generator.py`
  - `brainpy_state/_nest/mip_generator.py`
  - `brainpy_state/_nest/pulsepacket_generator.py`
  - `brainpy_state/_nest/noise_generator.py`
  - `brainpy_state/_nest/spike_generator.py`
  - `brainpy_state/_nest/spike_train_injector.py`
  - `brainpy_state/_nest/spike_dilutor.py`
  - `brainpy_state/_nest/multimeter.py`
  - `brainpy_state/_nest/spike_recorder.py`
  - `brainpy_state/_nest/weight_recorder.py`
  - `brainpy_state/_nest/correlation_detector.py`
  - `brainpy_state/_nest/correlomatrix_detector.py`
  - `brainpy_state/_nest/correlospinmatrix_detector.py`
  - `brainpy_state/_nest/spin_detector.py`
- Read (catalog reference): `docs/nest-status/internal/nest-catalog-snapshot.md` (device sections)

**Scope:** Stimulation generators (current, rate, spike, Poisson variants, sinusoidal, gamma, MIP, pulsepacket, noise), spike injectors, dilutors, recorders (multimeter, spike_recorder, weight_recorder), detectors (correlation, correlomatrix, correlospinmatrix, spin).

- [ ] **Step 1: Enumerate ported devices.**

```bash
grep -E "from \.[a-z_0-9]*(generator|recorder|detector|multimeter|dilutor|injector|volume_transmitter) import" brainpy_state/_nest/__init__.py | \
  sed 's/.*from \.\([a-z_0-9]*\) import.*/\1/' | sort -u
```

- [ ] **Step 2: For each lead device, extract semantics.**

Capture per file:
- Init signature: device-specific parameters with units.
- What the device emits/records on each step (currents? spikes? voltage traces?).
- Recording cadence (every step? every `interval`? on-spike-only?).
- Output buffering: does the recorder accumulate or emit-and-forget?
- Whether the device supports the NEST `record_from` list (per-state-variable subset) — multimeter especially.
- Whether spike generators support precise-spike-time semantics (`*_ps` variants).

- [ ] **Step 3: Fetch upstream docs per lead device.**

```
WebFetch url=https://nest-simulator.readthedocs.io/en/stable/models/<device_name>.html
        prompt="List parameters with defaults/units, what the device outputs, recording cadence, supported record_from variables (if recorder), and any documented limitations."
```

Repeat for every lead.

- [ ] **Step 4: Diff and classify.**

Pay special attention to:
- **Recording-device fidelity** — `nest-status/index.rst` already self-discloses semantic divergence here. This task should make the divergence concrete: list the specific behaviors that differ.
- **Generator emit semantics** — `start`/`stop` windows, `origin` offsets, `frequency`/`amplitude` schedules.
- **Spike-time precision** — `_ps` variants.

- [ ] **Step 5: Compose `devices-gap.md` using the spec §2.3 template.**

Mapping table: one row per upstream NEST device. Status almost universally `unvalidated` or `partial` per the self-disclosure; mark concretely from the diff.

Semantic-risk section must explicitly call out:
- **Recording-device semantics divergence** — list each concrete behavior difference with file:line.
- **Generator timing window semantics** (`origin`, `start`, `stop`).
- **Precise-spike-time variants** — what `_ps` means in NEST vs. in the repo.

- [ ] **Step 6: Verify §4 evidence rules + acceptance criteria.**

Same as Task 2 Steps 6-7.

- [ ] **Step 7: Commit.**

```bash
git add docs/nest-status/internal/devices-gap.md
git commit -m "docs(nest-gap): add devices gap doc"
```

---

## Task 5: Network API gap doc

**Files:**
- Create: `docs/nest-status/internal/network-api-gap.md`
- Read: `brainpy_state/__init__.py`, `brainpy_state/_base.py`, plus relevant brainstate connection-management modules accessible from the Python env (`python -c "import brainstate; print(brainstate.__file__)"` then inspect siblings).
- Read (catalog reference): `docs/nest-status/internal/nest-catalog-snapshot.md` (PyNEST API + connection rules + spatial sections)

**Scope:** PyNEST top-level functions (`Connect`, `Create`, `CopyModel`, `GetStatus`/`SetStatus`, `GetConnections`, `NodeCollection`, `ResetKernel`, `SetKernelStatus`, etc.), all connection rules, spatial/topology layer, parameter expressions, model registry.

- [ ] **Step 1: Confirm NEST-API absence in `brainpy_state`.**

```bash
grep -rE "^def (Connect|Create|CopyModel|GetStatus|SetStatus|GetConnections|ResetKernel|SetKernelStatus|GetKernelStatus|NodeCollection)\b" brainpy_state/ 2>/dev/null | grep -v _test
```

Expected: no matches. The NEST PyNEST API is genuinely absent from `brainpy_state`; all `Connect()` references in the repo live in test comparison harnesses.

If any matches appear, capture them — they're real coverage that must be classified rather than declared missing.

- [ ] **Step 2: Catalog brainstate's connectivity primitives.**

```bash
python -c "import brainstate; help(brainstate.nn)" 2>&1 | grep -iE "connect|project|sparse|fixed|all_to_all" | head -40
```

Inspect `brainstate.nn` for: projection classes, connection-mask utilities, sparse-matrix builders. Note what's available — this is what an `align-to-NEST` shim would build on.

Also inspect `brainpy_state/_brainpy/projection.py` for the in-repo projection layer.

- [ ] **Step 3: Map each NEST PyNEST function onto the repo equivalent (or `missing`).**

For each PyNEST function in the catalog snapshot, fill one row in the mapping table. Status will be predominantly `missing` for the top-level API, `divergent` for primitives that exist with different shape (e.g., brainstate's projections vs. NEST's `Connect`).

- [ ] **Step 4: Compose `network-api-gap.md` using the spec §2.3 template.**

Mapping table structure (one row per PyNEST function or connection rule):

```markdown
| NEST API | Status | brainpy.state equivalent | NEST upstream | Tests | Notes |
|---|---|---|---|---|---|
| `Connect(pre, post, conn_spec={"rule":"all_to_all"}, syn_spec=...)` | missing | `brainstate.nn.AlignPostProj` (different shape; see brainpy-style projections) | <url> | none | NEST-style Connect is genuinely absent; brainpy.state expects users to compose projections directly. |
| `CopyModel(existing, new, params)` | missing | none | <url> | none | No model-registry / parameter-override mechanism. |
| `fixed_indegree` rule | missing | partial — sparse-mask utility in brainstate | <url> | none | <gap detail> |
| ... | | | | | |
```

Semantic & numerical risks section must call out:
- **Programming-model gap**: NEST is an imperative simulator API (`Create` → `Connect` → `Simulate`); brainpy.state is a JAX/brainstate compositional model. Either build a thin NEST-compat shim, or document the porting pattern.
- **Connection-rule parity**: which rules have any analog at all, which don't.
- **NodeCollection vs. brainpy populations**: indexing semantics, slicing, iteration.
- **Parameter expressions** (NEST's runtime-evaluated parameter trees) — absent.
- **Spatial/topology layer**: entirely absent.

Roadmap P0 candidates:
- A NEST-compat shim package or guide. Acceptance: a Brunel network can be expressed in NEST-like Python and run on brainpy.state with explicit translation rules documented.
- Connection-rule primitives that match NEST semantics (`fixed_indegree`, `pairwise_bernoulli`). Acceptance: spec'd primitives with tests.

- [ ] **Step 5: Verify §4 evidence rules + acceptance criteria.**

Same as Task 2 Steps 6-7. Note: for `missing` rows, the `brainpy.state location` column may legitimately be empty or contain `n/a`; the upstream link is still required.

- [ ] **Step 6: Commit.**

```bash
git add docs/nest-status/internal/network-api-gap.md
git commit -m "docs(nest-gap): add network API gap doc"
```

---

## Task 6: Examples gap doc

**Files:**
- Create: `docs/nest-status/internal/examples-gap.md`
- Read: `docs/examples/gallery.rst`
- Reference: upstream NEST PyNEST examples directory.

**Scope:** PyNEST example portfolio vs. `docs/examples/`. Goal: a porting-target list.

- [ ] **Step 1: Inventory the current repo examples.**

```bash
cat docs/examples/gallery.rst
find docs/examples/ -type f
find . -path ./brainpy_state -prune -o -name "*.ipynb" -print 2>/dev/null | grep -v node_modules | head -40
```

Expected: only `gallery.rst`. Note any tutorial notebooks in `docs/quickstart/` or `docs/brainpy-guide/` that mention NEST.

- [ ] **Step 2: Inventory upstream NEST examples.**

```
WebFetch url=https://nest-simulator.readthedocs.io/en/stable/examples/index.html
        prompt="List every example/tutorial linked from this page. For each, give the title, the script filename (e.g., brunel_alpha_nest.py), one-line description, and which model families it exercises."
```

If the index page links to subpages, fetch sub-indexes (e.g., plasticity examples, multimeter examples).

Alternative source: fetch `https://api.github.com/repos/nest/nest-simulator/contents/pynest/examples` to list example filenames directly.

- [ ] **Step 3: Build the porting-targets table.**

For each upstream example, decide:
- Is it feasible to port given current `brainpy_state/_nest/` coverage? (Cross-reference the models the example exercises against neurons-gap.md / synapses-plasticity-gap.md / devices-gap.md.)
- Priority: P0 (flagship — Brunel, microcircuit, balanced random network), P1 (common — plasticity demo, multimeter recording, Poisson input), P2 (niche).

- [ ] **Step 4: Compose `examples-gap.md` using the spec §2.3 template.**

Adapt the per-axis template:
- §3 mapping table: one row per upstream example, status ∈ {`implemented` (ported and working), `partial` (ported but missing pieces), `missing` (not ported), `unsupported` (depends on NEST features out of scope per spec §7)}.
- §4: missing examples — list them.
- §5: semantic risks — pre-flag examples where porting will surface model gaps (e.g., a multimeter-heavy example will trip on recording-device divergence).
- §7 roadmap: P0 should include at least Brunel and microcircuit.

Example row format:

```markdown
| brunel_alpha_nest.py | missing | n/a | https://nest-simulator.readthedocs.io/en/stable/auto_examples/brunel_alpha_nest.html | none | Flagship benchmark; requires iaf_psc_alpha + Poisson generator + multimeter; all three present but unvalidated. |
```

Roadmap P0 candidate:
- **Port Brunel network as flagship example.** [L]
  Acceptance: `docs/examples/brunel.ipynb` reproduces NEST's firing-rate result within 5% over 1s with matched seeds and matched dt. Test script lives at `brainpy_state/_nest/_validation/brunel_test.py` (skipped by default; opt-in via env var).

- [ ] **Step 5: Verify §4 evidence rules + acceptance criteria.**

Same as Task 2 Steps 6-7.

- [ ] **Step 6: Commit.**

```bash
git add docs/nest-status/internal/examples-gap.md
git commit -m "docs(nest-gap): add examples portfolio gap doc"
```

---

## Task 7: Docs portfolio gap doc

**Files:**
- Create: `docs/nest-status/internal/docs-portfolio-gap.md`
- Read: `docs/index.rst`, `docs/api/index.rst`, `docs/api/nest-*.rst`, `docs/nest-status/index.rst`, `docs/quickstart/`, `docs/brainpy-guide/`.
- Reference: upstream NEST user docs TOC.

**Scope:** NEST user-doc tiers vs. repo's `docs/` tree. Identify tier-level gaps (most notably: no `docs/nest-guide/` porting tutorial).

- [ ] **Step 1: Inventory repo docs.**

```bash
find docs/ -type f \( -name "*.rst" -o -name "*.md" -o -name "*.ipynb" \) | grep -v _build | sort
```

Capture the structure into a tree-style listing.

- [ ] **Step 2: Inventory upstream NEST docs TOC.**

```
WebFetch url=https://nest-simulator.readthedocs.io/en/stable/index.html
        prompt="List every top-level section in the documentation TOC (Getting started, Models, Networks, Running simulations, Recording, etc.). For each section, list its child pages (one level deep) with one-line descriptions."
```

Tier candidates likely to appear: Installation, Tutorials, Models reference, Connection management, Recording from simulations, Stimulating networks, Random numbers in NEST, Parallel computing, Examples, Reference (PyNEST API), Glossary.

- [ ] **Step 3: Tier-by-tier diff.**

For each NEST doc tier, identify the repo equivalent (or absence). Examples of likely findings:
- **Getting started**: repo has `docs/quickstart/` — `implemented`.
- **Models reference**: repo has `docs/api/nest-neurons.rst` etc. — but those are auto-generated API stubs, not narrative model-reference pages with parameter tables. → `partial`.
- **Connection management guide**: no equivalent → `missing`.
- **Recording from simulations guide**: no equivalent → `missing`.
- **Random numbers guide**: no equivalent → `missing`.
- **Parallel computing**: intentionally divergent (JAX device sharding instead of MPI) → `divergent` (or `unsupported` if no JAX-equivalent guide planned).
- **PyNEST API reference**: absent → `missing` (but follows from `network-api-gap.md`).
- **NEST porting guide**: absent → `missing` (P0 candidate).

- [ ] **Step 4: Compose `docs-portfolio-gap.md` using the spec §2.3 template.**

Mapping table: one row per NEST doc tier with status, repo location (or empty), upstream link, notes.

Roadmap P0 candidates:
- **Add `docs/nest-guide/` — porting tutorial from PyNEST → brainpy.state.** [L]
  Acceptance: tutorial covers (a) creating neurons, (b) creating generators, (c) connecting populations, (d) recording, (e) simulation; each step shown side-by-side as PyNEST code + brainpy.state equivalent.
- **Add a parameter-table convention** for the NEST model docs in `docs/api/nest-neurons.rst` — each model lists every parameter with default, unit, NEST upstream link. [M] Acceptance: parameter tables present for at least the IAF and AdEx families.

- [ ] **Step 5: Verify §4 evidence rules + acceptance criteria.**

Same as Task 2 Steps 6-7.

- [ ] **Step 6: Commit.**

```bash
git add docs/nest-status/internal/docs-portfolio-gap.md
git commit -m "docs(nest-gap): add docs portfolio gap doc"
```

---

## Task 8: Numerical validation gap doc

**Files:**
- Create: `docs/nest-status/internal/numerical-validation-gap.md`
- Read: every `brainpy_state/_nest/*_test.py` for `import nest` references.

**Scope:** Cross-cutting. Per-family inventory of NEST-comparison test coverage. Lists families that need a NEST-comparison harness before promotion from Experimental → Beta.

- [ ] **Step 1: Inventory NEST-comparison tests.**

```bash
cd brainpy_state/_nest
for f in *_test.py; do
  if grep -q "^import nest\|^from nest\| nest\." "$f" 2>/dev/null; then
    echo "HAS_NEST $f"
  else
    echo "NO_NEST  $f"
  fi
done | sort > /tmp/nest_test_inventory.txt
wc -l /tmp/nest_test_inventory.txt
grep -c "^HAS_NEST" /tmp/nest_test_inventory.txt
grep -c "^NO_NEST" /tmp/nest_test_inventory.txt
```

Capture the two counts and the full inventory.

- [ ] **Step 2: For each `HAS_NEST` test, characterize the comparison.**

`Read` each `HAS_NEST` test file enough to capture:
- What quantity is compared (membrane potential trace, spike times, firing rate, weight trajectory).
- Tolerance (`atol`, `rtol`, or implicit).
- Duration and dt.
- Whether the test is run by default in CI or skipped behind a flag.

- [ ] **Step 3: Family-level rollup.**

Group findings by family (IAF psc, IAF cond, AdEx, GIF, GLIF, HH, MAT, rate, binary, point-process, multi-compartment, astrocyte, STDP, STP, Clopath, Urbanczik, devices, generators, recorders, detectors, gap junctions, etc.). For each family compute: `n_tests`, `n_with_nest`, `n_skipped`, coverage %.

- [ ] **Step 4: Identify families with no NEST-comparison coverage.**

These families are the validation-harness priorities. Each becomes a roadmap item.

- [ ] **Step 5: Compose `numerical-validation-gap.md` using the spec §2.3 template.**

Adapt the per-axis template:
- §3 mapping table: one row per family. Columns: family, `n_tests`, `n_with_nest`, coverage %, representative compared quantity, tolerance, notes.
- §4: families with zero coverage.
- §5: numerical risks. Specifically call out: (a) PRNG divergence (bit-exact impossible), (b) integration-step coupling (NEST uses fixed dt + ring buffers; brainpy.state uses adaptive RKF45 or analytical propagators), (c) refractory-period rounding to dt grid.
- §6: validation-infrastructure gaps. Is there a reusable `nest_compare()` harness? Where would it live?
- §7: roadmap. P0 = build the harness + cover IAF + AdEx; P1 = HH + STDP; P2 = devices/recorders/correlators.

Reusable harness P0 acceptance criterion:
- **P0 — Build `brainpy_state/_nest/_validation/nest_compare.py`.** [M]
  A reusable helper that runs a `brainpy.state` neuron and an upstream NEST neuron with matched parameters/seeds/dt for N ms and returns (repo_trace, nest_trace, max_abs_diff, max_rel_diff). Skipped under `pytest -m "not requires_nest"` by default. Acceptance: at least 3 IAF and 3 AdEx tests use it and pass within tolerance documented in the harness.

- [ ] **Step 6: Verify §4 evidence rules + acceptance criteria.**

Same as Task 2 Steps 6-7.

- [ ] **Step 7: Commit.**

```bash
git add docs/nest-status/internal/numerical-validation-gap.md
git commit -m "docs(nest-gap): add numerical validation gap doc"
```

---

## Task 9: Index document with consolidated roadmap

**Files:**
- Create: `docs/nest-status/internal/index.md`
- Read (all per-axis docs from Tasks 2-8):
  - `docs/nest-status/internal/neurons-gap.md`
  - `docs/nest-status/internal/synapses-plasticity-gap.md`
  - `docs/nest-status/internal/devices-gap.md`
  - `docs/nest-status/internal/network-api-gap.md`
  - `docs/nest-status/internal/examples-gap.md`
  - `docs/nest-status/internal/docs-portfolio-gap.md`
  - `docs/nest-status/internal/numerical-validation-gap.md`

- [ ] **Step 1: Extract bucket counts from each per-axis doc.**

For each per-axis doc, copy its §2 parity-summary counts.

- [ ] **Step 2: Extract every P0 and P1 roadmap item from each per-axis doc.**

Grep:

```bash
grep -A 2 "^- \*\*P0" docs/nest-status/internal/*-gap.md
grep -A 2 "^- \*\*P1" docs/nest-status/internal/*-gap.md
```

Capture the full text of each into a working list. Each item already carries its T-shirt size and acceptance criterion per the per-axis template — preserve them verbatim.

- [ ] **Step 3: Order the consolidated roadmap.**

Apply the spec §6 prioritization principles. Default ordering heuristic:

1. P0 from `numerical-validation-gap.md` (the harness — blocks everything else).
2. P0 from `network-api-gap.md` (programming-model shim).
3. P0 from `docs-portfolio-gap.md` (porting guide — blocks user adoption).
4. P0 from `devices-gap.md` (recording-device fidelity — already self-disclosed).
5. P0 from `neurons-gap.md` (per-family validation).
6. P0 from `synapses-plasticity-gap.md`.
7. P0 from `examples-gap.md` (Brunel/microcircuit).
8. All P1 items in the same family order.

Deviate from this default only if the per-axis sweep surfaced something more urgent — and if you deviate, leave a one-line `<!-- ordering note: ... -->` HTML comment explaining why.

- [ ] **Step 4: Get current git SHA for the header.**

```bash
git rev-parse HEAD
```

- [ ] **Step 5: Compose `index.md`.**

Constraint: under ~150 lines.

Template:

```markdown
# NEST Parity Gap Analysis — Internal Maintainer Index

**Last updated:** YYYY-MM-DD
**Git SHA at analysis:** <sha>
**NEST reference version:** 3.x (latest stable on nest-simulator.readthedocs.io)
**Audience:** brainpy.state maintainers. Not built into the public Sphinx site.

This index rolls up the seven per-axis gap analyses in this directory. The
authoritative spec for methodology, taxonomy, and evidence rules is at
`../../superpowers/specs/2026-05-11-nest-gap-analysis-design.md`. Each per-axis
doc owns its own evidence table; this index owns the consolidated roadmap.

## Parity summary

| Axis | implemented | unvalidated | partial | divergent | missing | unsupported | Doc |
|---|---|---|---|---|---|---|---|
| Neurons                | N | N | N | N | N | N | [neurons-gap.md](neurons-gap.md) |
| Synapses & plasticity  | N | N | N | N | N | N | [synapses-plasticity-gap.md](synapses-plasticity-gap.md) |
| Devices                | N | N | N | N | N | N | [devices-gap.md](devices-gap.md) |
| Network API            | N | N | N | N | N | N | [network-api-gap.md](network-api-gap.md) |
| Examples               | N | N | N | N | N | N | [examples-gap.md](examples-gap.md) |
| Docs portfolio         | N | N | N | N | N | N | [docs-portfolio-gap.md](docs-portfolio-gap.md) |
| Validation coverage    | N | N | N | N | N | N | [numerical-validation-gap.md](numerical-validation-gap.md) |

## Headline findings

1. <one-sentence finding, e.g., "Most ported neurons are `unvalidated` — present and structurally NEST-compatible but lacking numerical comparison against NEST reference traces.">
2. <one-sentence finding, e.g., "The NEST PyNEST API surface is essentially absent — there is no `Connect`, `CopyModel`, or NodeCollection equivalent.">
3. <one-sentence finding>
4. <one-sentence finding>

## Consolidated roadmap

### P0 (blocks family promotion or credible porting)

1. **<item title>** [<size>] — <source doc>
   Acceptance: <criterion>.
2. **<item title>** [<size>] — <source doc>
   Acceptance: <criterion>.
... (one numbered entry per P0 from per-axis docs, in the ordered list from Step 3)

### P1 (parameter drift, common variants, flagship examples)

1. **<item title>** [<size>] — <source doc>
   Acceptance: <criterion>.
...

### P2 (edge cases, polish)

Summarized at index level — see per-axis docs for full list.

## Intentionally unsupported

These are out of scope by design (see spec §7). Not gaps:

- MPI / multi-process distribution
- MUSIC interface
- NESTML and SLI modeling languages
- Real-time / hardware-in-the-loop devices
- Bit-exact RNG parity (distributional parity is in scope)
- NEST kernel internals (event scheduler, ring buffers, slice scheduling)

## Methodology and classification reference

- Methodology, taxonomy, and evidence rules: see
  `../../superpowers/specs/2026-05-11-nest-gap-analysis-design.md` §3-5.
- Status values used in mapping tables: `implemented`, `unvalidated`, `partial`,
  `divergent`, `missing`, `unsupported`. Defined in spec §3.
- Catalog snapshot the analysis is diffed against:
  [nest-catalog-snapshot.md](nest-catalog-snapshot.md).
```

- [ ] **Step 6: Verify the consolidated roadmap is a strict superset of the per-axis P0/P1 lists.**

```bash
for f in docs/nest-status/internal/*-gap.md; do
  echo "=== $f ==="
  grep -E "^- \*\*(P0|P1)" "$f" | head -20
done
```

Cross-check each per-axis bullet appears in `index.md` under the right priority section.

- [ ] **Step 7: Verify the file is under ~150 lines.**

```bash
wc -l docs/nest-status/internal/index.md
```

Expected: ≤ 200 (the soft cap is 150; allow up to 200 if the roadmap is large, but tighten if larger).

- [ ] **Step 8: Commit.**

```bash
git add docs/nest-status/internal/index.md
git commit -m "docs(nest-gap): add consolidated index + roadmap"
```

---

## Task 10: Final acceptance-criteria pass

**Files:**
- All under `docs/nest-status/internal/` (no new files unless gaps surfaced).

- [ ] **Step 1: Run the spec §8 acceptance checklist.**

For each item in spec §8, verify:

```bash
# 1. All 7 per-axis docs + index exist.
ls docs/nest-status/internal/{index,neurons-gap,synapses-plasticity-gap,devices-gap,network-api-gap,examples-gap,docs-portfolio-gap,numerical-validation-gap}.md
# 2. Evidence rules — every row has repo path + NEST upstream link (where applicable).
for f in docs/nest-status/internal/*-gap.md; do
  echo "=== $f ==="
  grep -c "nest-simulator.readthedocs.io" "$f"
  grep -c "brainpy_state/_nest/" "$f"
done
# 3. Catalog snapshot has version + date.
head -10 docs/nest-status/internal/nest-catalog-snapshot.md
# 4. Index roadmap superset (verified in Task 9 Step 6).
# 5. Every roadmap item has "Acceptance:".
for f in docs/nest-status/internal/*-gap.md docs/nest-status/internal/index.md; do
  n_items=$(grep -cE "^- \*\*P[012]" "$f")
  n_accept=$(grep -c "Acceptance:" "$f")
  echo "$f items=$n_items accept=$n_accept"
done
# 6. conf.py excludes internal dir.
grep "nest-status/internal" docs/conf.py
```

For any failed check, fix the offending doc in place and amend with a follow-up commit.

- [ ] **Step 2: Render a final summary in the terminal.**

Bullet what was produced, what's classified in which bucket per axis, and the top 3 P0 items pulled from `index.md`. This is what the agent reports back at the end.

- [ ] **Step 3: Final commit (if any fixes were needed).**

```bash
git add docs/nest-status/internal/
git status
# If anything changed in Step 1 fixes:
git commit -m "docs(nest-gap): final acceptance-criteria fixes"
```

If nothing changed, skip the commit.

---

## Plan self-review notes

Coverage cross-check vs. spec:
- Spec §2.1 layout (index + catalog snapshot + 7 per-axis docs): Tasks 1, 2, 3, 4, 5, 6, 7, 8, 9. ✓
- Spec §2.2 index content (header, parity table, consolidated roadmap, unsupported list, methodology reference): Task 9 Step 5 template. ✓
- Spec §2.3 per-axis template: Tasks 2-8 each apply it. ✓
- Spec §2.4 axis assignments: 1-to-1 with Tasks 2-8. ✓
- Spec §3 classification taxonomy: enforced via mapping-table status column in each per-axis task. ✓
- Spec §4 evidence rules: Step 6 in Tasks 2-4, Step 5 in Task 5, Step 5 in Tasks 6-8, plus Task 10 Step 1 final check. ✓
- Spec §5 methodology: Tasks 1 (catalog snapshot), 2-8 (read 15-20 leads + diff + classify). ✓
- Spec §6 prioritization: Tasks 2-8 §7 roadmap section + Task 9 Step 3 ordering. ✓
- Spec §7 unsupported list: Task 9 Step 5 index template. ✓
- Spec §8 acceptance criteria: Task 10 Step 1 checklist. ✓

No placeholders. Every step contains the actual command, the actual template, or the actual file lists.

Type/name consistency: status values `{implemented, unvalidated, partial, divergent, missing, unsupported}` are used identically across all tasks. T-shirt sizes `{S, M, L, XL}` are used identically. Priority tiers `{P0, P1, P2}` are used identically.
