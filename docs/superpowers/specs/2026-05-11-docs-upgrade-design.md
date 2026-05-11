# brainpy.state Documentation Upgrade — Design Spec

**Date:** 2026-05-11
**Owner:** Documentation upgrade workstream
**Status:** Approved scope; ready for implementation plan

---

## 1. Background and goal

`brainpy.state` documentation today does not distinguish between the stable
BrainPy-style modeling API (45+ models, frozen public interface for the 0.0.x
series) and the experimental NEST-compatible API (119+ models, mid-refactor on
the `update` branch: `brainunit` → `saiunit`, scalar loops → vectorized
`AdaptiveRungeKuttaStep`, numpy state arrays → `DotDict` PyTrees, Python
exceptions → `brainstate.transform.jit_error_if`). The README, the docs landing
page, and every NEST API reference page implicitly present 167+ models as
production-ready.

**Goal:** upgrade the documentation so that:

1. New users immediately see what is stable and what is experimental.
2. The BrainPy-style modeling APIs are documented as complete and recommended
   for production work.
3. The NEST-compatible APIs are clearly labeled experimental, with a dedicated
   status page that explains scope, the in-flight refactor, and what users
   should not yet rely on.
4. The README accurately reflects the project, the canonical docs URL, and the
   maturity split.
5. Contributors have explicit guidance on documenting new APIs and marking
   experimental features.

**Non-goals (for this round):**

- Per-NEST-model maturity triage. All NEST models receive a blanket
  Experimental label.
- Rewriting the existing notebooks (`5min-tutorial.ipynb`, `architecture.ipynb`,
  `neurons.ipynb`, `synapses.ipynb`, `projections.ipynb`).
- Adding NEST-style examples to the gallery.
- Migrating documentation tooling. Sphinx + `sphinx_book_theme` is retained.

---

## 2. Documentation audit — issues found (prioritized)

### Critical

| # | Issue | Location |
|---|---|---|
| C1 | NEST-style APIs are presented as stable; no maturity callout anywhere | `README.md`, `docs/index.rst`, `docs/api/nest-*.rst` |
| C2 | README links to `brainpy-state.readthedocs.io`; the canonical site is `brainx.chaobrain.com/brainpy-state/` (per commit `90f8523`) | `README.md` |
| C3 | No dedicated NEST status page — users have no way to learn what is incomplete or what may change | (does not exist) |

### High

| # | Issue | Location |
|---|---|---|
| H1 | `docs/core-concepts/` covers only BrainPy-style concepts, but the section name implies framework-wide scope | `docs/core-concepts/` |
| H2 | API reference does not group Stable vs Experimental in navigation | `docs/index.rst` toctree, `docs/api/index.rst` |
| H3 | No `.. warning::` admonition on any of the four `nest-*.rst` API pages | `docs/api/nest-*.rst` |
| H4 | `CONTRIBUTING.md` does not address how to document new APIs or how to mark experimental features | `CONTRIBUTING.md` |
| H5 | README quickstart snippet is incomplete (4 lines, no simulation, no projection) | `README.md` |

### Medium

| # | Issue | Location |
|---|---|---|
| M1 | Examples gallery has no NEST-style examples and no maturity labeling on entries | `docs/examples/gallery.rst` |
| M2 | No FAQ / Troubleshooting page beyond the install subsection | (does not exist) |
| M3 | `CLAUDE.md` references `upgrade.md` in the repo root, but the file does not exist | `CLAUDE.md`, repo root |
| M4 | `5min-tutorial.ipynb` is a 117 KB single notebook — hard to scan and maintain | `docs/quickstart/5min-tutorial.ipynb` |

### Low

| # | Issue | Location |
|---|---|---|
| L1 | `docs/conf.py` `intersphinx_mapping` pins Python 3.8 docs | `docs/conf.py` |
| L2 | README does not include a License section line | `README.md` |
| L3 | `docs/quickstart/installation.rst` "Verifying Installation" snippet uses `brainpy.__version__` rather than `brainpy_state.__version__` | `docs/quickstart/installation.rst` |

This round addresses **all Critical and all High items**, plus L2 and L3 as
cheap fixes. Medium and the remaining Low items are tracked as follow-ups in
§9.

---

## 3. Approach

**Approach A — inline labels + banners (selected).**

- File structure of `docs/` is preserved except for one folder rename
  (`core-concepts/` → `brainpy-guide/`, per §4) and one new folder
  (`nest-status/`).
- Maturity is communicated through three mechanisms:
  1. A top-of-page `.. admonition:: API maturity` block on `docs/index.rst`.
  2. A `.. warning::` admonition at the top of every NEST API reference page.
  3. A separate "Experimental (NEST-Compatible)" toctree caption that groups the
     new NEST status page with the four NEST API reference pages.
- The README receives a structured rewrite including an API maturity table.

Rejected: Approach B (hard split into a top-level Experimental section). It
would require relocating files and would break inbound deep-links to existing
NEST API pages without a corresponding benefit beyond what Approach A delivers
via the toctree caption.

---

## 4. Information architecture

### Target `docs/` tree

```
docs/
├── index.rst                          # MODIFIED: maturity callout, restructured toctrees
├── quickstart/
│   ├── index.rst                      # MODIFIED: update cross-ref to brainpy-guide/
│   ├── installation.rst               # MODIFIED: fix __version__ reference (L3)
│   ├── overview.ipynb                 # unchanged
│   └── 5min-tutorial.ipynb            # unchanged this round
├── brainpy-guide/                     # RENAMED from core-concepts/
│   ├── index.rst                      # MODIFIED: retitle to "BrainPy-style Modeling Guide"
│   ├── architecture.ipynb             # unchanged
│   ├── neurons.ipynb                  # unchanged
│   ├── synapses.ipynb                 # unchanged
│   └── projections.ipynb              # unchanged
├── examples/
│   └── gallery.rst                    # unchanged this round
├── api/
│   ├── index.rst                      # MODIFIED: Stable vs Experimental sections
│   ├── base.rst                       # unchanged
│   ├── brainpy-*.rst                  # unchanged (7 files)
│   ├── nest-base.rst                  # MODIFIED: warning banner
│   ├── nest-neurons.rst               # MODIFIED: warning banner
│   ├── nest-synapses.rst              # MODIFIED: warning banner
│   ├── nest-plasticity.rst            # MODIFIED: warning banner
│   └── nest-devices.rst               # MODIFIED: warning banner
├── nest-status/                       # NEW
│   └── index.rst                      # NEW
├── changelog.md
└── conf.py                            # unchanged this round (L1 deferred)
```

### Toctree caption restructure in `docs/index.rst`

Replace the current two toctree blocks with four:

- `Tutorials` — `quickstart/index.rst`, `brainpy-guide/index.rst`,
  `examples/gallery.rst`
- `API Reference (Stable)` — `api/base`, `api/brainpy-neurons`,
  `api/brainpy-synapses`, `api/brainpy-projections`, `api/brainpy-synouts`,
  `api/brainpy-plasticity`, `api/brainpy-readouts`, `api/brainpy-inputs`
- `Experimental (NEST-Compatible)` — `nest-status/index`, `api/nest-base`,
  `api/nest-neurons`, `api/nest-synapses`, `api/nest-plasticity`,
  `api/nest-devices`
- `Project` — `changelog.md`

### Folder rename — `core-concepts/` → `brainpy-guide/`

Rationale: the directory only contains BrainPy-style concept notebooks
(`architecture`, `neurons`, `synapses`, `projections`). Naming it
`core-concepts/` implies framework-wide scope, which misleads readers about
NEST-style coverage.

**Tradeoff accepted:** inbound deep-links to
`brainx.chaobrain.com/brainpy-state/core-concepts/*` will return 404 until the
docs site rebuilds. No external sites are known to link to these pages
(documentation is still pre-1.0). The rename happens once and is cheap to
revert.

**Cross-reference updates required:**

- `docs/index.rst` grid card and toctree
- `docs/quickstart/index.rst` — two `:doc:` references to `core-concepts/index`
- `docs/quickstart/installation.rst` — one `:doc:` reference to
  `../core-concepts/index`
- Any further internal references discovered via
  `grep -r "core-concepts" docs/`

---

## 5. README rewrite

### Target structure

```
1. Header (badges + logo)              [keep as-is]
2. One-paragraph description           [tighten current text]
3. Key Features                        [trim and clarify]
4. API Status & Maturity               [NEW — maturity table]
5. Installation                        [keep current tabs]
6. Quickstart Example                  [REWRITE — runnable, BrainPy-style only]
7. Documentation                       [NEW — correct URL]
8. Examples & Tutorials                [NEW — link to gallery]
9. Development Status                  [NEW — brief]
10. Contributing                       [NEW — link to CONTRIBUTING.md]
11. Ecosystem                          [keep current table]
12. Citation                           [keep current]
13. License                            [NEW — single line referencing LICENSE]
```

### API Status & Maturity table (Section 4)

| Component | Status | Notes |
|---|---|---|
| Base classes (`Dynamics`, `Neuron`, `Synapse`) | **Stable** | Public API stable for the 0.0.x series |
| BrainPy-style neurons (LIF, ALIF, HH, Izhikevich, …) | **Stable** | 45+ models, fully tested |
| BrainPy-style synapses, projections, readouts, inputs | **Stable** | COBA / CUBA / MgBlock, STP / STD, Expon / Alpha / AMPA / GABA / NMDA |
| Surrogate-gradient training | **Stable** | All BrainPy-style neurons differentiable |
| NEST-compatible neurons (IAF, AdEx, GIF, GLIF, HH, …) | **Experimental** | 60+ models under active development; parameter names, defaults, and numerical behavior may change |
| NEST-compatible synapses & plasticity (STDP, STP) | **Experimental** | Under active development |
| NEST-compatible devices (generators, recorders, detectors) | **Experimental** | Under active development |
| Rate / binary neurons (NEST set) | **Experimental** | Subset of NEST set; coverage incomplete |

### Quickstart snippet

The new quickstart snippet must be a runnable BrainPy-style example. The
implementing change pulls the structure from `examples/103_COBA_2005.py` so
the README mirrors real, tested code rather than fabricating signatures.

The snippet must:

- Import `brainstate`, `brainpy.state as bps`, `saiunit as u`.
- Set `brainstate.environ.set(dt=0.1 * u.ms)`.
- Define a small `EINet` (`brainstate.nn.DynamicalSystem`) with `bps.LIF`
  populations and at least one projection.
- Show one simulation step or a short `brainstate.compile.for_loop`.
- Stay under ~25 lines.

The exact text is produced during implementation by inspecting
`examples/103_COBA_2005.py`. No fabricated function names.

### URL fixes

- `https://brainpy-state.readthedocs.io/` → `https://brainx.chaobrain.com/brainpy-state/`
- Documentation link points to `…/brainpy-state/index.html`
- Quickstart link points to `…/brainpy-state/quickstart/5min-tutorial.html`
- Examples link points to `…/brainpy-state/examples/gallery.html`

### Length and tone

Target ~150-180 lines. Maintain the existing professional tone. No marketing
language. No claims of production-readiness for NEST-style.

---

## 6. `docs/index.rst` maturity callout and NEST API banners

### Maturity callout — inserted under the opening paragraph in `docs/index.rst`

```rst
.. admonition:: API maturity
   :class: important

   ``brainpy.state`` ships two model families with different maturity levels:

   - **Stable** — Base classes and **BrainPy-style models** (45+ neurons,
     synapses, projections, readouts, input generators). Public API is stable
     for the 0.0.x series and recommended for production use.
   - **Experimental — In Development** — **NEST-compatible models** (119+
     neurons, synapses, plasticity, devices). These are under active
     development and parameter names, defaults, and numerical behavior may
     change without notice. Use them for exploration and validation, but pin
     your dependency version and expect breaking changes.

   See the :doc:`NEST-style status page <nest-status/index>` for what is
   currently available and what users should not rely on yet.
```

### Banner — top of every `docs/api/nest-*.rst` page (including `nest-base.rst`)

```rst
.. warning::

   **Experimental — In Development.** The NEST-compatible model family is
   under active development. Parameter names, defaults, numerical behavior,
   and the set of available models may change without notice across 0.0.x
   releases. See the :doc:`NEST-style status page </nest-status/index>` for
   current scope and limitations.
```

Inserted **before** the page's existing top-level heading so it renders above
the title.

### `docs/api/index.rst` restructure

Replace the current single "Organization" section with two top-level
sub-headings:

- **Stable API** — contains the Base Classes card and the seven BrainPy-style
  cards (Neurons, Synapses, Projections, Synaptic Outputs, Short-Term
  Plasticity, Readouts, Input Generators).
- **Experimental API (NEST-Compatible)** — contains the NEST Base Classes
  card and the four NEST cards (Neurons, Synapses, Plasticity, Devices),
  preceded by a brief callout linking to the NEST status page.

Hidden toctrees at the bottom of the page mirror the four-caption
restructure described in §4.

---

## 7. NEST status page — `docs/nest-status/index.rst`

A single new file, ~200-300 lines. Sections:

1. **Page-level warning admonition** (same wording as the API page banner).
2. **What is the NEST-compatible model family?** — one-paragraph explanation
   of the goal: faithful JAX re-implementations of NEST neurons, synapses,
   plasticity, and devices with NEST-compatible parameter names, backed by
   `brainstate` and `saiunit`, JIT-compilable on CPU/GPU/TPU.
3. **What is currently available** — grouped by integration category from
   `CLAUDE.md`:
   - Category A — AdEx, GIF, GLIF, IAF (cond) — full RKF45 via
     `AdaptiveRungeKuttaStep`
   - Category B — IAF (psc) — exact analytical propagators
   - Category C — Hodgkin-Huxley family — `AdaptiveRungeKuttaStep`
   - Category D — Rate models — vectorized patterns
   - Category E — Devices, synapses, plasticity — no ODE integration

   The page links out to the existing `api/nest-*.rst` pages for model
   lists rather than duplicating them.
4. **Active migration in progress** — bullet list of in-flight refactor items
   (`brainunit` → `saiunit`, scalar loops → vectorized
   `AdaptiveRungeKuttaStep`, manual arrays → `DotDict`, validation rework,
   `jit_error_if` migration). Explains that APIs called today may behave
   subtly differently after the refactor lands.
5. **What users should not rely on yet** — explicit list:
   - Numerical equivalence with NEST (mostly unverified).
   - Stable parameter names and defaults (track NEST upstream but may shift).
   - Multi-compartment models (`_mc` suffix) — particularly experimental.
   - Recording device fidelity to NEST device semantics — partial.
   - Plasticity (STDP, STP) — learning behavior unvalidated against NEST
     reference traces.
6. **Recommended usage** —
   - Pin dependency version (`brainpy.state==0.0.4`).
   - Validate critical numerical behavior locally.
   - Prefer BrainPy-style for production and surrogate-gradient training.
   - Use NEST-compatible models when parameter parity for porting or
     comparison is needed.
   - Report mismatches via GitHub issues with a minimal reproducer.
7. **Roadmap** — kept high-level to avoid rot:
   - Complete `brainunit` → `saiunit` migration.
   - Vectorized ODE integration across all Category A/C models.
   - Per-family validation suites against NEST reference traces.
   - Promote families from Experimental to Beta as validation lands.
8. **See also** — cross-references to all four `api/nest-*.rst` pages and
   the upstream NEST simulator docs.

The page intentionally avoids per-model status entries (we chose blanket
Experimental in §3 and on the maturity table in §5).

---

## 8. `CONTRIBUTING.md` additions

Two new subsections appended to the existing "Documentation" section (or
inserted as their own top-level section if cleaner — implementer's call):

### 8.1 Documenting new APIs

- Every new public class or function must have a NumPy-style docstring with
  `Parameters`, `Returns` (or yields), and at least one `Examples` block.
- BrainPy-style models: add an `autosummary` entry to the appropriate
  `docs/api/brainpy-*.rst` page.
- NEST-compatible models: add an `autosummary` entry to the appropriate
  `docs/api/nest-*.rst` page; ensure parameter names match the upstream NEST
  documentation; document any deliberate deviation in the docstring under a
  "Parameter Mapping" or "Implementation Notes" section
  (already registered as `napoleon_custom_sections` in `docs/conf.py`).
- For any new model file, also add a colocated `*_test.py` that exercises
  default parameters and at least one parameter sweep.

### 8.2 Marking experimental features

- New NEST-compatible models inherit the family's experimental status — no
  per-model banner is required so long as the model lives under
  `brainpy_state/_nest/`.
- New BrainPy-style features that are not yet stable must include a Sphinx
  `.. warning::` admonition at the top of the docstring labelled
  **Experimental — In Development**, and must be omitted from the README
  maturity table until promoted to Stable.
- Promoting a feature from Experimental to Stable requires: (a) full test
  coverage including parameter validation; (b) removal of the warning
  admonition from the docstring; (c) addition to the README maturity table;
  (d) a CHANGELOG entry under "API stability".
- When in doubt, ship as Experimental. Promoting is cheap; demoting silently
  breaks users.

### 8.3 Documentation review checklist (for PR authors)

A short bullet list added to the existing PR checklist:

- [ ] Docstrings present on all new public APIs (NumPy style).
- [ ] New module exported via `__all__` in `brainpy_state/__init__.py`.
- [ ] Added to the corresponding `docs/api/*.rst` page.
- [ ] If NEST-compatible: parameter names match upstream NEST.
- [ ] If experimental: no claims of stability in docstrings or README.
- [ ] `pytest brainpy_state/` passes locally.

---

## 9. Out of scope (deferred follow-ups)

These are real issues from the audit (§2) but explicitly not addressed this
round:

- **M1** Adding NEST-style examples to the gallery and labeling existing
  entries by maturity.
- **M2** Adding a top-level FAQ / Troubleshooting page.
- **M3** Either restoring `upgrade.md` to the repo root or removing the
  reference from `CLAUDE.md`.
- **M4** Splitting `5min-tutorial.ipynb` into a sequence of smaller pages.
- **L1** Updating `docs/conf.py` `intersphinx_mapping` to Python 3.13.
- Per-NEST-model maturity triage (rejected in §3).
- Documentation tooling migration (Sphinx + `sphinx_book_theme` is kept).
- **Notebook follow-up:** `docs/brainpy-guide/synapses.ipynb` does not currently
  execute end-to-end. Cell 6 fails inside `saiunit.exprel` with
  `exprel requires a dimensionless "x" when "unit_to_scale" is not provided`
  during `AlignPostProj` / `COBA` initialisation. A KNOWN ISSUE banner was
  added to the notebook on 2026-05-11. Refresh the notebook once the
  in-flight `saiunit` / `brainstate` API refactor stabilises.

---

## 10. Acceptance criteria

The implementation is complete when:

1. `README.md` matches the structure in §5, including the maturity table and
   the corrected docs URL. The quickstart snippet has been executed against
   the installed package per §11 and produces output without error.
2. `docs/index.rst` contains the maturity callout from §6 and the four-caption
   toctree restructure from §4.
3. The folder `docs/core-concepts/` has been renamed to `docs/brainpy-guide/`
   and all in-tree cross-references have been updated (`grep -r "core-concepts" docs/`
   returns no matches except in build artifacts).
4. `docs/nest-status/index.rst` exists with all eight sections from §7.
5. Each of `docs/api/nest-base.rst`, `docs/api/nest-neurons.rst`,
   `docs/api/nest-synapses.rst`, `docs/api/nest-plasticity.rst`,
   `docs/api/nest-devices.rst` carries the warning banner from §6 above its
   top-level heading.
6. `docs/api/index.rst` is restructured into Stable and Experimental
   sub-headings per §6.
7. `CONTRIBUTING.md` contains the three subsections from §8.
8. `docs/quickstart/installation.rst` "Verifying Installation" snippet uses
   `brainpy_state.__version__` (fix L3).
9. README "License" line is present (fix L2).
10. `make html` (or the project's equivalent docs build) completes without
    new warnings introduced by these changes.
11. **Notebook and code-block verification.** Every Jupyter notebook touched
    or referenced by this round, and every executable code block in the
    modified RST/Markdown files, must be executed end-to-end and confirmed to
    run without errors. See §11 for the verification protocol.
12. The README quickstart snippet has been executed in a Python environment
    with `brainpy.state` installed and produced output (the snippet is
    runnable, not pseudocode).

---

## 11. Code execution and verification protocol

Documentation that ships broken code is worse than no documentation. Every
code-bearing document this round touches must be executed before the change
is considered complete.

### 11.1 Scope of verification

**Jupyter notebooks** — every cell must execute without error end-to-end.
This round touches the following notebooks via the folder rename and
maturity-callout cross-references:

- `docs/quickstart/overview.ipynb`
- `docs/quickstart/5min-tutorial.ipynb`
- `docs/brainpy-guide/architecture.ipynb` (renamed from `core-concepts/`)
- `docs/brainpy-guide/neurons.ipynb`
- `docs/brainpy-guide/synapses.ipynb`
- `docs/brainpy-guide/projections.ipynb`

**RST and Markdown code blocks** — every fenced or directive code block
(```` ```python ````, `.. code-block:: python`, `.. code-block:: bash`) in
files this round creates or modifies must be confirmed runnable:

- `README.md` — quickstart snippet, install commands.
- `docs/index.rst` — quickstart snippet inside grid cards (if any).
- `docs/quickstart/installation.rst` — verification snippet and install
  commands.
- `docs/nest-status/index.rst` — any code samples added.
- `docs/api/index.rst`, `docs/api/nest-*.rst` — banner blocks (RST only;
  no Python to execute, but the banner directive must render).

### 11.2 Execution method

For notebooks: use `jupyter nbconvert --to notebook --execute --inplace
<path>` or equivalent. The cell outputs may be regenerated; that is
expected and acceptable for this round.

For Python code blocks in RST/Markdown: extract to a temporary `.py` file
and run it with `python tmpfile.py` in an environment that has
`brainpy.state` installed (`pip install -e ".[dev,cpu]"` per `CLAUDE.md`).

For shell code blocks (install commands): static review only — do not
execute `pip install` commands against the user's environment.

### 11.3 Failure handling

If a notebook or code block fails:

1. **Do not silently fix and ship.** Report the failure with the cell
   number, the error message, and the file path.
2. If the failure is due to an unrelated upstream bug (e.g., the
   `update`-branch refactor changed an API the notebook relied on), open
   a separate issue and add a TODO comment at the top of the affected
   notebook noting the broken cell. The notebook is then **excluded**
   from this round's acceptance and listed in §9 as a follow-up.
3. If the failure is due to the README quickstart snippet specifically,
   the snippet must be fixed before merge — the README is the most
   visible piece of documentation and cannot ship broken.

### 11.4 Evidence

The implementation report must include, for each executed notebook and
extracted code block:

- File path
- Pass/fail
- For failures: cell number, error excerpt, and disposition (fixed,
  deferred with issue link, or excluded)

---

## 12. Risks

- **Broken inbound links** from the `core-concepts/` rename. Mitigation: the
  rename is single-shot and reversible; the docs site is still pre-1.0; the
  change is announced in the next changelog entry.
- **README quickstart snippet** could fabricate signatures if the implementer
  doesn't actually read `examples/103_COBA_2005.py`. Mitigation: §5
  explicitly requires the snippet to mirror that file.
- **Toctree restructure** may cause Sphinx to warn about orphaned documents
  if any cross-references are missed. Mitigation: build the docs locally
  before commit; resolve any new warnings.
- **NEST status page becoming stale** as the migration lands. Mitigation: §7
  keeps roadmap items at high level; the page references categories defined
  in `CLAUDE.md` rather than per-model status.
