# brainpy.state Documentation Rebuild — Design Spec

- **Status:** Draft for review
- **Date:** 2026-06-17
- **Scope:** Complete, clean-slate rebuild of the `brainpy.state` documentation site (information architecture + content), keeping the Sphinx toolchain.
- **Owner:** Chaoming Wang
- **Companion file:** an implementation plan will be written separately (`docs-rebuild-plan.md`) after this spec is approved.

---

## 1. Background & motivation

`brainpy.state` is the point-neuron modeling layer of the BrainX ecosystem. It ships **two model families**, both production-ready:

- **BrainPy-style** — 45+ composable neurons, synapses, the AlignPre/AlignPost projections, synaptic outputs (COBA/CUBA/MgBlock), short-term plasticity, readouts, input generators. Differentiable via surrogate gradients.
- **NEST-compatible** — JAX re-implementations preserving NEST parameter names: ~65 neurons, 13 synapses, 14 plasticity rules, 27 devices, 7 spatial, plus base/network/validation modules. 75 ported example scripts. Numerically validated against a live NEST install within documented tolerance bands (see §10).

The current docs grew organically (`quickstart/`, `brainpy-guide/`, `../../examples`, `apis/`, `nest-guide/`, `nest-status/`). They do not (a) tell the project's defining story — *one differentiable substrate bridging brain simulation and brain-inspired computing* — nor (b) give the distinctive **AlignPre/AlignPost** design the prominence it deserves, nor (c) surface the live-NEST validation/parity evidence that substantiates trusting the NEST-compatible family.

This rebuild fixes all three.

### Defining narrative (the thesis the docs must argue)

The two anchoring papers:

1. Wang, C., Zhang, T., He, S., Gu, H., Li, S., Wu, S. (2024). *A differentiable brain simulator bridging brain simulation and brain-inspired computing.* ICLR 2024. <https://openreview.net/forum?id=AU2gS9ut61> (arXiv:2311.05106). — introduces AlignPre/AlignPost.
2. Wang, C., et al. (2026). *Model-agnostic linear-memory online learning in spiking neural networks.* Nature Communications. <https://doi.org/10.1038/s41467-026-68453-w> — shows that neuron-aligned synaptic state (AlignPre/AlignPost) is what enables linear-memory online learning (RTRL O(H³)/O(H²) → O(H)).

The docs treat AlignPre/AlignPost as the **keystone** that connects the two worlds: the same alignment that makes simulation memory-efficient is what makes gradient-based/online learning memory-efficient.

---

## 2. Goals & non-goals

### Goals

- A coherent information architecture (Approach C below) that serves a **dual audience** (computational neuroscience ↔ brain-inspired computing / SNN-ML) through one bridging narrative.
- A prominent, rigorous **Core Concepts** spine, centered on the AlignPre/AlignPost design and its link to differentiability and online learning.
- A rich, **self-contained NEST-compatible hub** (modeled on NEST's get-started page) carrying its unique porting / divergence / validation content.
- Surface the **live-NEST validation/parity evidence** (per-family parity status + tolerance categories) so users can see exactly why the NEST-compatible family is trustworthy.
- Every code-bearing page is a runnable, doctest-clean Jupyter notebook using the **public `brainpy.state`** surface end users actually import.

### Non-goals (YAGNI)

- No toolchain migration (stay on Sphinx; no Jupyter Book / Quarto).
- No duplication of upstream NEST model physiology — link out instead.
- No duplication of NEST's model *semantics* — the nest-like hub links out to upstream NEST for physiology (see §10).
- No bilingual build in this pass (English primary; revisit later if desired).
- No new library features — this is documentation only.

---

## 3. Locked decisions (from brainstorming)

| Axis | Decision |
|------|----------|
| Audience & positioning | **Dual audience, bridging narrative.** AlignPre/AlignPost is the keystone connecting brain simulation and brain-inspired computing. |
| Architecture | **Approach C** — Diátaxis discipline + a Core-Concepts spine + model-family split. |
| NEST strategy | **Full self-contained hub** (modeled on NEST get-started) + unique porting/divergence/validation content (§10). |
| Maturity labels | **None.** Both families presented as stable; validation evidence shown instead of badges (§16 D6). |
| Toolchain | **Stay on Sphinx.** Code-bearing pages → `.ipynb`; pure-prose pages → `.rst`/`.md`. |
| Language | **English primary.** |
| Rewrite scope | **Clean-slate** structure & content; salvage high-value assets (see §11). |

---

## 4. Constraints & conventions

These are hard constraints drawn from `../../CLAUDE.md` and must be honored in every doc page.

### 4.1 Namespace (CLAUDE.md rule #9)

- **All documentation example code uses the public `brainpy.state` namespace** (`from brainpy import state as ...`, or `from brainpy.state import ...`, or `brainpy.state.X`). Docs are end-user-facing.
- Docs example code **must not** import private `brainpy_state._*` submodules. If an example needs something internal, re-derive it from the public API.
- (Internal source/tests keep importing `brainpy_state` directly — not relevant to docs prose, but relevant if any doc references implementation.)

### 4.2 Driving models (CLAUDE.md rule #10)

- Example code **never** drives a model with a bare Python `for`/`while` loop when it runs repeatedly. Use `brainstate.transform`:
  - single step / one-shot → `jit`
  - many steps, collect outputs → `for_loop`
  - many steps with explicit carry → `scan`
  - long rollout under autograd → `checkpointed_for_loop` / `checkpointed_scan`
- The `state-paradigm` concept page and the `train-long-rollouts-checkpoint` how-to teach this explicitly.

### 4.3 Docstring & example style (CLAUDE.md "Docstring style")

- API reference is autodoc over NumPy-style docstrings (already the source convention).
- Notebook example blocks should be runnable and, where shown as expected output, doctest-consistent.

### 4.4 No maturity labels — show validation evidence instead

- **Both families are presented as stable/production-ready. The docs carry no "Stable"/"Experimental" badges, banners, or maturity table** (project decision, §16 D6).
- In place of fear-based labels, the docs **substantiate trust with evidence**: the `validation-status` page shows per-family live-NEST parity results and tolerance categories (A–E).
- API reference is organized by family for navigation, not by maturity.

### 4.5 Workflow (CLAUDE.md rules #6, #7)

- Implementation happens on a **worktree branch**, not `main`.
- This spec + the forthcoming plan live under `` (gitignored), not committed to repo history.

---

## 5. Information architecture — Approach C

**Approach C = Diátaxis discipline + a Core-Concepts (Explanation) spine + model-family split.**

Rationale (why C over the alternatives):

- **Rejected A (pure Diátaxis four-quadrant):** clean, but gives no natural home to the two model families or the bridging concept spine; AlignPre/AlignPost would be scattered across quadrants instead of getting one authoritative chapter.
- **Rejected B (two-worlds structural split):** strongest story but forces duplication — most building blocks (neurons, synapses, units) serve both worlds, so "where does this page go" becomes ambiguous and content is duplicated.
- **Chosen C:** keeps Diátaxis's "every page has one job," foregrounds AlignPre/AlignPost in a concept spine, gives each model family a clean organizational home, and lets the two-worlds narrative live where it is natural — in the **concepts** and at the **task (how-to) level** — without a structural split that breeds duplication.

Seven Sphinx toctree groups:

1. **Get Started** — install → run → mental model.
2. **Core Concepts (Explanation spine)** — where the bridging thesis is argued; AlignPre/AlignPost keystone.
3. **BrainPy-style Modeling** — full Diátaxis (tutorials → how-to); two how-to tracks (simulation / training); model lookup served by the API reference.
4. **NEST-Compatible** — full self-contained hub: own tutorials, build-your-network guides, a link to its example gallery (under `../../examples`), plus the unique porting / divergence / validation content.
5. **API Reference** — autodoc; organized by model family.
6. **Examples** — two galleries (brainpy-style + NEST), tagged by world + difficulty.
7. **Project** — changelog, citing, contributing, ecosystem.

**Deliberate asymmetry (D9).** BrainPy-style is the *native* layer: it leans on the shared Group 1 (Get Started) and Group 2 (Core Concepts) spine — AlignPre/AlignPost is a brainpy-style projection concept and lives there. NEST-compatible serves a distinct *migrant* audience entering from NEST, so Group 4 is a **self-contained hub** with its own onboarding that links *up* into the shared spine. Names stay `brainpy-style/` (idiomatic) and `nest-like/` (NEST look-alike).

---

## 6. Full file tree (target `..`)

```
docs/
├── index.rst                                   [prose]   Bridging hero + nav cards + ecosystem
│
├── get-started/                                ── Group 1 ──
│   ├── index.rst                               [prose]
│   ├── installation.rst                        [prose]   CPU/GPU/TPU backends; ecosystem install
│   ├── 5-minute-tour.ipynb                     [code]    Build + run an E-I net; spike raster (the "aha")
│   └── mental-model.ipynb                      [code]    4 big ideas; "two worlds, one substrate" seed
│
├── concepts/                                   ── Group 2: Core Concepts (spine) ──
│   ├── index.rst                               [prose]   The bridging thesis, reading order
│   ├── state-paradigm.ipynb                    [code]    State, init_all_states, environ.context, transform loops
│   ├── physical-units.ipynb                    [code]    brainunit; unit-safe construction & initializers
│   ├── model-anatomy.ipynb                     [code]    Dynamics → Neuron/Synapse; get_spike / current I/O
│   ├── alignpre-alignpost.ipynb       ★        [code]    KEYSTONE (see §8)
│   ├── differentiability.ipynb                 [code]    Surrogate grads + BPTT via transform (see §9.1)
│   └── online-learning.rst                     [prose]   Linear-memory RTRL + braintrace pointer (see §9.2)
│
├── brainpy-style/                              ── Group 3: BrainPy-style ──
│   ├── index.rst                               [prose]
│   ├── tutorials/
│   │   ├── 01-first-neuron.ipynb               [code]
│   │   ├── 02-synapse-and-projection.ipynb     [code]
│   │   ├── 03-ei-balanced-network.ipynb        [code]
│   │   └── 04-train-an-snn.ipynb               [code]
│   └── howto/
│       ├── index.rst                           [prose]
│       ├── sim-choose-neuron.ipynb             [code]    ┐ Simulation track
│       ├── sim-coba-cuba-synapses.ipynb        [code]    │
│       ├── sim-short-term-plasticity.ipynb     [code]    │
│       ├── sim-delays.ipynb                    [code]    │
│       ├── sim-reproduce-a-paper.ipynb         [code]    ┘ (gamma oscillation / Joglekar propagation)
│       ├── train-surrogate-gradients.ipynb     [code]    ┐ Training / brain-inspired track
│       ├── train-readouts.ipynb                [code]    │
│       └── train-long-rollouts-checkpoint.ipynb[code]    ┘ checkpointed_for_loop / scan
│
├── nest-like/                                  ── Group 4: NEST-Compatible (self-contained hub) ──
│   ├── index.rst                               [prose]   The hub: 3-tier card grid (modeled on NEST get-started)
│   ├── tutorials/                              ─ Tier 1: First steps ─
│   │   ├── 01-one-neuron.ipynb                 [code]    One NEST-compat neuron (iaf_psc_alpha): drive + record
│   │   ├── 02-populations.ipynb                [code]    Populations + devices
│   │   ├── 03-connect-network.ipynb            [code]    NEST-style wiring → small Brunel
│   │   └── 04-record-and-analyze.ipynb         [code]    Recorders/detectors; analyze output
│   ├── models.rst                              [prose]   Tier 2: model directory → API; link OUT to NEST for physiology
│   ├── connectivity.rst                        [prose]   Tier 2: connection rules + synapse spec (brainpy.state-flavored)
│   ├── devices.rst                             [prose]   Tier 2: generators / recorders / detectors (27 device models)
│   ├── spatial.rst                             [prose]   Tier 2: spatially-structured networks (_nest_spatial, 7)
│   ├── porting-walkthrough.ipynb               [code]    Tier 3: Brunel, side-by-side NEST → brainpy.state
│   ├── divergences/
│   │   ├── index.rst                           [prose]   Tier 3: semantic-divergence catalog
│   │   └── stdp.rst                            [prose]   Tier 3: SALVAGE existing stdp-divergences.rst
│   ├── validation-status.rst                   [prose]   Tier 3: live-NEST parity + A–E tolerance showcase
│   └── integration-categories.rst             [prose]   Tier 3: under-the-hood A–E ODE solving
│       (index features a "large networks" card → gallery entries; no separate page)
│
├── apis/                                       ── Group 5: API Reference ──
│   ├── index.rst                               [prose]   Organized by family (BrainPy-style / NEST-compatible)
│   ├── base.rst / brainpy-neurons.rst / brainpy-synapses.rst / brainpy-projections.rst /
│   │   brainpy-synouts.rst / brainpy-plasticity.rst / brainpy-readouts.rst / brainpy-inputs.rst  [autodoc]
│   └── nest-base.rst / nest-neurons.rst / nest-synapses.rst / nest-plasticity.rst /
│       nest-devices.rst / nest-spatial.rst / nest-network.rst                                   [autodoc]
│
├── examples/                                   ── Group 6 ──
│   ├── brainpy-gallery.rst                     [prose]   BrainPy-style examples (20 brainpy_like scripts)
│   └── nest-gallery.rst                        [prose]   NEST-compatible examples (75 nest_like scripts; NEST→brainpy.state map)
│
└── project/                                    ── Group 7 ──
    ├── changelog.md                            (existing)
    ├── citing.rst                              [prose]   eLife BrainPy + ICLR 2024 (no 2026 in cite-this block)
    └── ecosystem.rst                           [prose]   brainstate / brainunit / brainevent / braintools
```

Notes:
- `conf.py`, `_static/`, `_templates/`, `highlight_lexer.py`, `Makefile`, `make.bat` are retained/adjusted, not rewritten.
- `contributing` links to repo `../../CONTRIBUTING.md` (no duplication).

---

## 7. Chapter-by-chapter specification

Each entry: **path** — *format* — intent — content — source.

### Group 1 · Get Started

- **`get-started/installation.rst`** — *prose* — get a working install fast. Content: Python ≥3.10; `pip install brainpy.state[cpu|cuda12|cuda13|tpu]`; full-ecosystem `pip install BrainX`; verifying the install; backend notes. Source: salvage from `quickstart/installation.rst` + README install block.
- **`get-started/5-minute-tour.ipynb`** — *code* — the "aha." Content: build the EI COBA net from the README, `init_all_states`, run with `environ.context(dt=...)` + `transform.for_loop`, plot a spike raster. Audience: both. Source: README quickstart + `../../examples/brainpy_like/103_COBA_2005.py`.
- **`get-started/mental-model.ipynb`** — *code* — the 4 big ideas: (1) State-based programming, (2) physical units, (3) compose neurons + synapses + projections, (4) drive with `transform`. Closes with "two worlds, one substrate" framing and links into Core Concepts. Source: new; re-cut from `brainpy-guide/architecture.ipynb`.

### Group 2 · Core Concepts (the spine)

- **`concepts/index.rst`** — *prose* — states the bridging thesis explicitly, gives recommended reading order, and previews the keystone chapter.
- **`concepts/state-paradigm.ipynb`** — *code* — why functional JAX needs explicit State; `State`, `init_all_states`, `environ.context(dt, t)`; **the transform primitives** (`jit`/`for_loop`/`scan`/`checkpointed_*`) and the rule: never drive a model with a bare Python loop (CLAUDE.md #10). Source: new; re-cut from `brainpy-guide/architecture.ipynb`.
- **`concepts/physical-units.ipynb`** — *code* — `brainunit` quantities (`mV`, `ms`, `nS`, …), unit-safe construction, unit-aware initializers, common pitfalls caught at construction time. Source: new.
- **`concepts/model-anatomy.ipynb`** — *code* — the `Dynamics → Neuron / Synapse` base hierarchy; state variables; `update()`; `get_spike()`; how a neuron receives current (`add_current_input` / `add_delta_input`). Sets up the projection contract. Source: new; re-cut from `brainpy-guide/neurons.ipynb` + `synapses.ipynb`.
- **`concepts/alignpre-alignpost.ipynb`** ★ — *code* — the keystone. See §8.
- **`concepts/differentiability.ipynb`** — *code* — the *simulator → trainable* half of the bridge. See §9.1.
- **`concepts/online-learning.rst`** — *prose* — the *trainable → online / scalable* half. See §9.2.

### Group 3 · BrainPy-style Modeling

- **`brainpy-style/index.rst`** — *prose* — orient the reader; "start with tutorials, reach for how-to by task, look up models in the API reference."
- **Tutorials** (learning-oriented, sequential, *code*):
  - `01-first-neuron.ipynb` — instantiate a `LIFRef`, inject current, inspect V and spikes.
  - `02-synapse-and-projection.ipynb` — wire two populations with one `AlignPostProj`; observe PSCs.
  - `03-ei-balanced-network.ipynb` — the canonical COBA E-I net end to end. Source: `../../examples/brainpy_like/103_COBA_2005.py`.
  - `04-train-an-snn.ipynb` — surrogate-gradient training with a readout on (Fashion-)MNIST. Source: `examples/brainpy_like/201/202`.
- **How-to** (task-oriented, two tracks, *code* except index):
  - `howto/index.rst` — *prose* — splits Simulation vs Training tracks.
  - Simulation: `sim-choose-neuron`, `sim-coba-cuba-synapses`, `sim-short-term-plasticity`, `sim-delays`, `sim-reproduce-a-paper` (gamma oscillation / Joglekar). Source: `examples/brainpy_like/107–113, 204`.
  - Training: `train-surrogate-gradients`, `train-readouts`, `train-long-rollouts-checkpoint` (checkpointed loops for BPTT memory). Source: `examples/brainpy_like/200–203`.

### Group 4 · NEST-Compatible (self-contained hub)

See §10 for the full hub design — a NEST-get-started-style 3-tier hub: **Tier 1** tutorials (`tutorials/01-04`), **Tier 2** build-your-network guides (`models`, `connectivity`, `devices`, `spatial`) plus a link to the NEST gallery (`examples/nest-gallery.rst`), **Tier 3** porting & fidelity (`porting-walkthrough`, `divergences/`, `validation-status`, `integration-categories`).

### Group 5 · API Reference

- **`apis/index.rst`** — *prose* — organize cards by family (BrainPy-style / NEST-compatible) for navigation; add `nest-spatial` and `nest-network` cards to cover those families.
- Autodoc pages: `base` + `brainpy-*` and `nest-*`, grouped by family; add `nest-spatial.rst`, `nest-network.rst`.

### Group 6 · Examples

- **`examples/brainpy-gallery.rst`** — *prose* — the **brainpy-style** examples (20 `brainpy_like` scripts), tagged **world** (simulation / training) and **difficulty**. Source: rename/extend the existing `examples/gallery.rst`.
- **`examples/nest-gallery.rst`** — *prose* — the **NEST-compatible** examples: all 75 `nest_like` scripts catalogued as a NEST→brainpy.state Rosetta map. Linked from the nest-like hub (Tier 2 + the "large networks" card). Source: `../../examples/nest_like`.

### Group 7 · Project

- **`project/changelog.md`** — keep existing `../../changelog.md`.
- **`project/citing.rst`** — *prose* — the eLife BrainPy + ICLR 2024 BibTeX (the cite-this-software block; **no 2026 Nature Comms here** per D12). The 2026 paper appears as a related-work reference inside `online-learning.rst`, not in this block.
- **`project/ecosystem.rst`** — *prose* — brainstate / brainunit / brainevent / braintools.

---

## 8. Keystone chapter spec — `concepts/alignpre-alignpost.ipynb`

This is the centerpiece. Audience: both, but written so a deep-learning reader and a neuroscience reader each find their footing.

### 8.1 Internal section flow

1. **The synaptic-state explosion.** Naive per-synapse state is **O(p · N_pre · N_post)**. Concrete: 10 000 → 10 000 at 10% density ≈ 10⁸ floats for ~10⁷ real synapses. Memory & compute waste.
2. **The key observation.** *"All synaptic variables sharing the same linear dynamics can be reduced to a single one."*
3. **Decoupling dynamics from communication — the four roles:** `comm` (connection matrix / DL layer), `syn` (synapse dynamics), `out` (COBA / CUBA / MgBlock), `post` (target population). → **Diagram 1**.
4. **AlignPost.** State aligned to **post**, **O(N_post)**, computed *after* comm. The exponential-synapse update `g ← g + 1` per arriving spike (presynaptic-identity-agnostic) → contributions sum linearly → one merged postsynaptic state. **Exponential-family only.** Natural for **many-to-one** fan-in; event-driven. Worked example: the COBA E-I net (`AlignPostProj` + `Expon.desc` + `COBA.desc`, from the README/`103_COBA_2005.py`).
5. **AlignPre.** State aligned to **pre**, **O(N_pre)**, computed *before* comm. Homogeneous per-presynaptic-neuron parameters → all synapses from one presynaptic neuron share one trace. **Works for all synapse models**, including nonlinear AMPA (`dg/dt = α[T](1−g) − βg`) and NMDA. Natural for **one-to-many** fan-out. Worked example: NMDA + `MgBlock`, or one population broadcasting to several targets (`align_pre_projection` / `CurrentProj`).
6. **Exact, not approximate.** The conditions: AlignPre needs homogeneous synaptic parameters per presynaptic neuron; AlignPost needs linear (exponential) dynamics (superposition). Cite: *"these projections are not approximations. They accurately compute the same dynamics as the original projections."*
7. **Automatic merging.** Multiple projections sharing post + synapse/output descriptor (AlignPost) or pre + delay (AlignPre) share a single synapse instance.
8. **Choosing between them** — the decision table (§8.2). → **Diagram 2**.
9. **The bridge to learning.** Neuron-aligned synaptic state is exactly what makes online learning linear-memory — forward-link to `online-learning.rst` (§9.2) and the 2026 Nature Comms result.
10. **API pointers** — `AlignPostProj`, `align_pre_projection`, `align_post_projection`, `CurrentProj`, `DeltaProj`, gap-junction projections.

### 8.2 Decision table (draft, to refine in implementation)

| Want | Synapse type | Connectivity | Use | State |
|------|-------------|-------------|-----|-------|
| Event-driven fan-in | Exponential-family (Expon, DualExpon) | many→one | **AlignPost** (`AlignPostProj`) | O(N_post) |
| Nonlinear kinetics | AMPA, NMDA, GABAa | any | **AlignPre** (`align_pre_projection`) | O(N_pre) |
| One source → many targets | any | one→many | **AlignPre** | O(N_pre) |
| Direct current / no synapse dynamics | — | any | `CurrentProj` / `DeltaProj` | — |

### 8.3 Complexity summary table (draft)

| Scheme | State lives on | State memory | Applies to | Best for |
|--------|----------------|--------------|------------|----------|
| Naive per-synapse | each synapse | O(p · N_pre · N_post) | any | (wasteful) |
| **AlignPre** | presynaptic neurons | O(N_pre) | **all** synapse models | one-to-many |
| **AlignPost** | postsynaptic neurons | O(N_post) | exponential-family only | many-to-one |

### 8.4 Key quotes to embed (with attribution)

- ICLR 2024: *"these projections are not approximations. They accurately compute the same dynamics as the original projections while providing new benefits."*
- ICLR 2024: *"with homogeneous synaptic parameters, the spike train coming from the same presynaptic neuron will lead to the same synaptic dynamics … all synapses originating from the same presynaptic neuron can share a single dynamical variable … AlignPre … is suitable for all types of synapse models."*
- ICLR 2024: *"the applicability of AlignPost is limited to synapse models that exhibit exponential dynamics … the conductance g of a post-synaptic neuron is updated according to g ← g+1 whenever a spike arrives and regardless of which presynaptic neuron emitted this spike."*
- Nature Comms 2026: *"BrainScale leverages the sparse, event-driven nature of spiking neurons to reformulate real-time recurrent learning (RTRL), which achieves linear memory scaling relative to the number of neurons."*

### 8.5 Terminology note to surface

The simulator is **BrainPy** / the modern API is **brainpy.state**; the online-learning engine is **BrainScale** (preprint) / **BrainTrace** (published; the `braintrace` package). One sentence in `online-learning.rst` prevents confusion.

---

## 9. Bridge chapters spec (the split — D11)

The bridge is **two pages**, because differentiability is runnable from brainpy.state, but online learning's engine lives in a separate ecosystem package (`braintrace`).

### 9.1 `concepts/differentiability.ipynb` — *code, runnable*

The *simulator → trainable* half. Native to brainpy.state (neurons take a `spk_fun` surrogate; the `train-*` examples run). Sections:

1. **Why spikes need surrogate gradients** — the non-differentiable threshold; `braintools.surrogate` functions plugged in via `spk_fun`.
2. **Backprop through time with `transform`** — `for_loop`/`scan` are differentiable; `checkpointed_*` bound memory for long rollouts.
3. **A small runnable trainable example** — a gradient step on a tiny SNN (links to the `train-*` how-to track).

### 9.2 `concepts/online-learning.rst` — *prose (conceptual + pointer)*

The *trainable → online / scalable* half. Not runnable from brainpy.state alone (the engine is the `braintrace` package), so this is explanation + cross-link with illustrative, non-executed snippets. Sections:

1. **From offline BPTT to online RTRL** — why RTRL, and its naive cost.
2. **Linear-memory RTRL** — the 2026 Nature Comms result. Because AlignPre/AlignPost align synaptic hidden state to a **neuron** dimension (not the weight dimension), and SNN dynamics are sparse/event-driven, RTRL's eligibility trace + parameter Jacobian (cubic) and hidden Jacobian (quadratic) collapse to **O(H)**. The same superposition that makes AlignPost's forward pass exact is reused on the backward pass (callback to the keystone, §8).
3. **Whole-brain scaling (D1 lands here)** — *Drosophila*-scale SNN as the payoff; link to the paper.
4. **The engine: `braintrace`** — terminology (BrainScale preprint → BrainTrace published) and a pointer to the `braintrace` package/docs. Illustrative snippet shown but not executed.

---

## 10. NEST-compatible hub design (detailed — D8)

A **self-contained hub modeled on NEST's get-started page** (a 3-tier card grid). Two guardrails: (a) **don't duplicate NEST's model *semantics*** — link out to upstream NEST for physiology; our differentiators are the brainpy.state API usage, porting parity, divergences, and the validation showcase; (b) the hub's tutorials are **brainpy.state-flavored** (State + units + `transform`), not copies of PyNEST.

**`nest-like/index.rst`** — the hub landing: a 3-tier card grid (mirrors NEST's "First steps / Build your own network / porting & fidelity"), plus a featured **"large networks"** card pointing into `examples/nest-gallery.rst` (Brunel, EI-clustered, astrocyte nets). Source: re-cut from `nest-guide/index.rst` + `nest-status/index.rst`.

**Tier 1 — First steps (tutorials, `[code]`):** mirror PyNEST Parts 1–4 in brainpy.state idiom.
- `tutorials/01-one-neuron.ipynb` — one NEST-compat neuron (`iaf_psc_alpha`), drive + record.
- `tutorials/02-populations.ipynb` — populations + devices.
- `tutorials/03-connect-network.ipynb` — NEST-style wiring → small Brunel.
- `tutorials/04-record-and-analyze.ipynb` — recorders/detectors; analyze output.

**Tier 2 — Build your own network (`[prose]`):**
- `models.rst` — model directory grouped by family, **linking to API and out to NEST** for physiology (no duplicated semantics).
- `connectivity.rst` — connection rules + synapse spec, brainpy.state-flavored.
- `devices.rst` — generators / recorders / detectors (27 device models).
- `spatial.rst` — spatially-structured networks (`_nest_spatial`, 7 modules).
- The **NEST gallery** lives at `examples/nest-gallery.rst` (all 75 `nest_like` scripts as a NEST→brainpy.state Rosetta map); the hub links to it from Tier 2 and the "large networks" card, rather than hosting its own copy. Source: `../../examples/nest_like`.

**Tier 3 — Porting & fidelity (our differentiators):**
- `porting-walkthrough.ipynb` `[code]` — Brunel side by side, NEST vs brainpy.state; parameter mapping + divergences encountered. Source: `examples/nest_like/brunel_*` + ``.
- `divergences/index.rst` — the **Semantic Divergences** catalog (where state lives, parameter-location maps, numerical bands, pairing conventions); every entry backed by live-NEST parity tests.
- `divergences/stdp.rst` — SALVAGE the existing `stdp-divergences.rst` verbatim (re-homed); highest-trust asset in the doc set.
- `validation-status.rst` — the trust showcase: per-family/per-model live-NEST parity results, tolerance categories (A–E from `tolerance_conventions.py`), how to run the harness (`pytest -m requires_nest`), known coverage gaps. Source: salvage/expand `nest-status/index.rst` + `_nest_validation/`.
- `integration-categories.rst` — under-the-hood A–E ODE solving (RKF45 / analytic propagators / vectorized rate updates / no-ODE devices). Source: salvage from `nest-status/index.rst`.

No maturity banners on these pages; trust is established by the validation evidence, not warnings.

---

## 10A. README & status-consistency refactor (D12, D13)

The new `..` carry no maturity labels (D6); these committed artifacts must be brought into line, or they will openly contradict the docs. (Grep confirmed the **package source and `pyproject.toml` carry no maturity language** — so the surface is just README + CONTRIBUTING; `docs/index.rst` is replaced anyway.)

### 10A.1 `../../README.md` target structure (D12)

| # | Section | Change |
|---|---------|--------|
| 1 | Header image + badges | keep |
| 2 | **Intro** | rewrite around the **bridging tagline** ("one differentiable substrate bridging brain simulation and brain-inspired computing"); two families, both production-ready |
| 3 | **Key features** | refresh; **add an AlignPre/AlignPost bullet** (distinctive memory-efficient projections) and "NEST parameter parity, validated against live NEST" |
| 4 | Installation | keep |
| 5 | **Quickstart** | **two labeled snippets** — a BrainPy-style one (keeps `AlignPostProj`, links the AlignPre/AlignPost concept) **and a NEST-compatible one**. GitHub markdown = sequential labeled blocks; docs landing uses real `tab-set`. (Q7 = option a.) |
| 6 | **Two model families** | **NEW** section replacing the "API status and maturity" table — BrainPy-style (native) vs NEST-compatible (validated), both stable, links to the validation showcase |
| 7 | Documentation | update links to the new IA (get-started / concepts / brainpy-style / nest-like / apis / examples) |
| 8 | Examples | keep, refresh |
| 9 | Contributing | keep (CONTRIBUTING itself fixed in 10A.2) |
| 10 | Ecosystem | keep; **add `braintrace`** (online-learning points to it) |
| 11 | Citing | **unchanged** — eLife + ICLR 2024 only; **do NOT add the 2026 paper** (owner decision) |
| 12 | License | keep |
| — | ~~API status and maturity~~ table | **removed** |
| — | ~~Development status~~ (stable/experimental) | **removed** |

README keeps its existing horizontal logo header (Diagram 3 is for the docs landing, not the README).

### 10A.2 `../../CONTRIBUTING.md` (D13)

Rewrite the **"Marking experimental features"** subsection (lines 238–268): drop the README-maturity-table reference, the Experimental→Stable promotion ceremony, and the "ship as Experimental when in doubt" rule. Replace with guidance to **document live-NEST parity/validation** for new NEST models (add a parity test under `_nest_validation/`, record the tolerance category, update `validation-status.rst`) and to keep docstrings claim-accurate.

### 10A.3 Left as-is

- `../../changelog.md` — append-only historical record; **not rewritten** (1 legacy mention is fine).
- `../../pyproject.toml` description — **optional** light touch to add the bridging tagline; skippable, low priority.

---

## 11. Old → new salvage / migration map

Clean-slate **structure**, but salvage high-value **content**:

| Existing | Disposition | New home |
|----------|-------------|----------|
| `quickstart/installation.rst` | salvage | `get-started/installation.rst` |
| `quickstart/5min-tutorial.ipynb` | salvage/re-cut | `get-started/5-minute-tour.ipynb` |
| `quickstart/overview.ipynb` | re-cut | `get-started/mental-model.ipynb` |
| `brainpy-guide/architecture.ipynb` | re-cut | `concepts/state-paradigm.ipynb` + `mental-model.ipynb` |
| `brainpy-guide/neurons.ipynb` | re-cut | `concepts/model-anatomy.ipynb` + tutorials |
| `brainpy-guide/synapses.ipynb` | re-cut | `concepts/model-anatomy.ipynb` + `sim-coba-cuba-synapses.ipynb` |
| `brainpy-guide/projections.ipynb` | **promote & expand** | `concepts/alignpre-alignpost.ipynb` (keystone) |
| `brainpy-guide/network-api.md` | fold | tutorials / how-to |
| `nest-guide/stdp-divergences.rst` | salvage verbatim | `nest-like/divergences/stdp.rst` |
| `nest-guide/index.rst` | re-cut | `nest-like/index.rst` + `nest-like/divergences/index.rst` |
| `nest-status/index.rst` | split | `nest-like/validation-status.rst` + `nest-like/integration-categories.rst` |
| `examples/nest_like/*` (75 scripts) | catalogue | `examples/nest-gallery.rst` (Rosetta map) |
| `apis/*` | keep, extend | `apis/*` (+ `nest-spatial`, `nest-network`) |
| `examples/gallery.rst` | rename & re-tag | `examples/brainpy-gallery.rst` (brainpy-style only) |
| `../../changelog.md` | keep | `project/changelog.md` |
| `../../README.md` | **refactor** (D12) | `../../README.md` (bridging narrative; maturity table removed) |
| `../../CONTRIBUTING.md` | **rewrite** experimental section (D13) | `../../CONTRIBUTING.md` |

---

## 12. Diagrams — specifications & GPT-image-2 prompts

Shared visual language (from the BrainScale figures): **orange = presynaptic (ℝ^N_pre)**, **blue = postsynaptic (ℝ^N_post)**, **green = connection matrix θ (ℝ^N_pre×N_post)**. Output target: SVG/PNG in `docs/_static/concepts/`. ⚠️ Image models render text imperfectly; expect label touch-up.

### Diagram 1 — AlignPre vs AlignPost (placement: §8.3 of keystone)

> A clean, flat, minimalist scientific vector diagram on a pure white background, two panels side by side separated by a thin vertical divider. Technical-illustration style, thin rounded strokes, soft drop shadows, a restrained palette of orange, blue, and green. LEFT panel titled "AlignPre": a vertical orange rounded rectangle on the left labeled "Presynaptic neurons (ℝ^N_pre)", then immediately to its right a small orange box labeled "Synapse dynamics" with a curved decay-trace icon inside, then a green trapezoid/matrix labeled "Connection (comm)", then a blue rounded rectangle labeled "Postsynaptic neurons (ℝ^N_post)" with a small "out: COBA/CUBA" tag. A horizontal arrow flows left→right through all blocks. A caption strip under the panel reads "State aligned to PRE · O(N_pre) · works for all synapses · one-to-many". RIGHT panel titled "AlignPost": same orange presynaptic rectangle, then the green connection matrix FIRST, then a blue "Synapse dynamics" box (now after the connection, colored blue), then the blue postsynaptic rectangle. Caption: "State aligned to POST · O(N_post) · exponential synapses · many-to-one". Emphasize that the synapse box sits BEFORE the green matrix on the left and AFTER it on the right. Label fonts clean sans-serif, no clutter, lots of whitespace.

### Diagram 2 — Memory complexity, three regimes (placement: §8.1/§8.2 of keystone)

> A clean flat scientific infographic on a white background, three labeled panels in a row, minimalist vector style with orange/blue/green palette and thin strokes. PANEL 1 "Naïve per-synapse": a dense full grid of tiny gray dots connecting an orange column of pre-neurons to a blue column of post-neurons, every crossing filled, with a big red label "O(p · N_pre · N_post)" and a small "✗ wasteful" tag. PANEL 2 "AlignPre": an orange column of pre-neurons, each pre-neuron carrying ONE small orange trace icon, with fan-out arrows spreading from each single trace to many blue post-neurons; label "O(N_pre)" in orange, tag "one-to-many". PANEL 3 "AlignPost": a blue column of post-neurons, each carrying ONE merged blue trace icon, with many orange pre-arrows converging (fan-in) into each single blue trace; label "O(N_post)" in blue, tag "many-to-one, event-driven". Consistent neuron glyphs (small circles) across panels, generous whitespace, clean sans-serif labels.

### Diagram 3 — The bridging concept (placement: docs landing hero + `online-learning.rst` §9.2; README keeps its existing logo header, not this)

> A wide, elegant conceptual banner on a white background, flat modern scientific style, soft gradients. Center-bottom: a broad horizontal slab labeled "One differentiable substrate" containing three small chips: "State", "Physical units", "AlignPre / AlignPost". Rising from this slab, two large rounded panels angle upward-left and upward-right like an open book. LEFT panel, cool blue tones, labeled "Brain Simulation": small icons of a biophysical network, a spike raster, conductance curves. RIGHT panel, warm orange tones, labeled "Brain-Inspired Computing": small icons of a trainable network, a loss curve going down, a gradient symbol ∇. A curved double-headed arrow arcs over the top connecting the two panels, labeled "surrogate gradients · linear-memory online learning". Minimalist, balanced, lots of whitespace, clean sans-serif type, suitable as a documentation hero image.

---

## 13. Cross-cutting writing conventions

- **Tone:** precise, researcher-to-researcher; no marketing fluff.
- **Every notebook:** opens with a one-paragraph "what you'll learn / who it's for (sim / training / both)"; imports `brainpy.state` (public namespace); drives models with `transform`; ends with "see also" links.
- **No maturity labels:** no Stable/Experimental badges or banners anywhere; the `validation-status` page carries the parity evidence instead (§16 D6).
- **Cross-links:** concepts link forward to how-to/tutorials; tutorials link back to concepts; API pages link to the catalog.
- **Citations:** ICLR 2024 anchors the keystone; the 2026 Nature Comms anchors `online-learning.rst`; `project/citing.rst` holds the cite-this-software BibTeX (**eLife + ICLR 2024 only — no 2026**).
- **One prose exception in the spine:** `concepts/online-learning.rst` is prose (not a notebook) because its engine (`braintrace`) is a separate package; illustrative snippets are shown but not executed.

---

## 14. Success criteria

1. A reader from either world can answer, after Get Started + Core Concepts: *what is the State paradigm, why units, and what are AlignPre/AlignPost and when do I use each.*
2. AlignPre/AlignPost has a single authoritative chapter with both diagrams and the decision table, citing both papers.
3. The bridging thesis is explicit in the landing page, `mental-model`, and the bridge chapter.
4. The NEST validation evidence (divergence catalog + per-family parity table) is findable in ≤2 clicks from the NEST index, substantiating the models' trustworthiness; no maturity banners anywhere.
5. All notebooks execute clean under the docs build (CI), using only the public `brainpy.state` API and `transform`-driven loops.
6. Nothing valuable from the old docs is silently dropped (see §11 map).

---

## 15. Implementation phasing (high-level; detailed plan to follow)

1. **Scaffold** — new `..` tree, toctree groups, `conf.py` adjustments, salvage moves (stdp-divergences, changelog, gallery), `api/`→`apis/`, redirects/stubs.
2. **Core Concepts spine** — `state-paradigm`, `physical-units`, `model-anatomy`, **alignpre-alignpost (keystone)**, **differentiability** + **online-learning** (split). (Highest value; do early.)
3. **Get Started** — installation, 5-minute-tour, mental-model.
4. **BrainPy-style** — tutorials, how-to (both tracks).
5. **NEST-like hub** — index (3-tier card grid), tutorials 01–04, models/connectivity/devices/spatial, porting-walkthrough, divergences (incl. salvage), validation-status, integration-categories.
6. **API + Examples + Project** — `apis/` autodoc (+ nest-spatial, nest-network), both galleries (`brainpy-gallery` + `nest-gallery`), citing/ecosystem.
7. **README + status consistency (§10A)** — refactor `../../README.md`; rewrite `CONTRIBUTING.md`'s experimental section.
8. **Diagrams** — generate via GPT-image-2 from §12 prompts; place in `_static/concepts/`; touch up labels.
9. **Build & verify** — full Sphinx build clean; notebook execution in CI; link check.

Each phase lands on a worktree branch (CLAUDE.md #6).

---

## 16. Decisions log

- **D1 — Whole-brain/scaling (superseded by D11):** now a section of `online-learning.rst` (its own page), not a folded subsection of a combined chapter. Engine still lives in the ecosystem (`braintrace`).
- **D2 — How-to structure:** **keep the two tracks** (Simulation / Training) split inside Group 3 — surfaces the dual audience at the task level.
- **D3 — Diagram prompts:** kept as proposed (§12).
- **D4 — Toolchain:** Sphinx retained; `.ipynb` for code, `.rst`/`.md` for prose (locked).
- **D5 — Language:** English primary; bilingual deferred (non-goal this pass).
- **D6 — No maturity labels (2026-06-17):** both BrainPy-style and NEST-compatible are presented as stable/production-ready. **All "Stable"/"Experimental" badges, banners, and the landing maturity table are removed.** Rationale: the NEST family has extensive passing live-NEST parity tests (131 validation files, formal A–E tolerance bands); trust is substantiated by *showing that evidence* on the `validation-status` page rather than by fear-based labels. (Author decision — overrides the earlier graduated-maturity proposal.)
- **D7 — README / package status consistency (now actioned — D12/D13):** confirmed the surface is **README + CONTRIBUTING only** (grep found no maturity language in package source or `../../pyproject.toml`); `docs/index.rst` is replaced anyway. Folded into this effort (§10A).
- **D8 — nest-like = full self-contained hub (Q1):** supersedes the earlier "porting-and-parity only" decision. Modeled on NEST's get-started page (3-tier card grid). Guardrails: no duplication of NEST model semantics (link out); tutorials are brainpy.state-flavored.
- **D9 — Keep the asymmetry (Q2) + names (Q3):** brainpy-style is the native layer on the shared Get-Started + Concepts spine (AlignPre/AlignPost lives there); nest-like is a self-contained migrant hub. Names: `brainpy-style/` + `nest-like/` (not symmetric `_like`).
- **D10 — Renames:** `docs/nest/` → `docs/nest-like/`; `docs/api/` → `../apis` (restores the repo's original plural).
- **D11 — Split the bridge chapter:** `concepts/differentiability.ipynb` (`[code]`, runnable — surrogate grads + BPTT) + `concepts/online-learning.rst` (`[prose]` — linear-memory RTRL + `braintrace` pointer). Format asymmetry because the online-learning engine is a separate package.
- **D12 — README refactor:** bridging narrative; remove the maturity table + "Development status"; add a "Two model families" section; **two quickstart snippets** (brainpy-style + nest-like); add `braintrace` to ecosystem; **citing unchanged (no 2026 paper)**. README keeps its existing logo header.
- **D13 — Status consistency:** rewrite `../../CONTRIBUTING.md`'s "Marking experimental features" section to match D6 (validation/parity guidance, no promotion ceremony); leave `changelog.md` (historical); `pyproject` description touch is optional/skippable.
- **D14 — Two galleries, co-located under `../../examples` (refined):** `examples/brainpy-gallery.rst` (20 `brainpy_like` scripts) and `examples/nest-gallery.rst` (75 `nest_like` scripts). The nest-like hub *links* to `nest-gallery` (Tier 2 + "large networks" card) rather than hosting its own copy. Each script listed once.
- **D15 — Drop `model-catalog.rst`:** the `apis/` autodoc pages already serve as the model directory; a separate narrative catalog is redundant. The `brainpy-style/index.rst` points readers to the API reference for model lookup.

---

## 17. Open questions (for review)

- Q1 — Are the 4 tutorials the right backbone, or do you want a 5th (e.g. a recurrent/reservoir or a custom-neuron tutorial)?
- Q2 — Should `examples/brainpy-gallery.rst` / `examples/nest-gallery.rst` pull thumbnails (matplotlib output) or stay text-only for now?
- Q3 — On the `validation-status` page, should families be listed by parity-tolerance category (A–E) or by model family as the primary axis? (Both are possible.)

---

## 18. References

1. Wang et al. (2024), ICLR — *A differentiable brain simulator bridging brain simulation and brain-inspired computing.* <https://openreview.net/forum?id=AU2gS9ut61> · arXiv:2311.05106.
2. Wang et al. (2026), Nature Communications — *Model-agnostic linear-memory online learning in spiking neural networks.* <https://doi.org/10.1038/s41467-026-68453-w>.
3. BrainScale preprint (bioRxiv 2024.09.24.614728) — richest prose on AlignPre/AlignPost alignment & dimension/color scheme.
4. Wang et al. (2023), eLife — *BrainPy …* <https://doi.org/10.7554/eLife.86365>.
5. NEST simulator documentation — <https://nest-simulator.readthedocs.io/>.
```
