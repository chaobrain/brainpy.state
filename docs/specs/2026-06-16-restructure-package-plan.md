# Restructure `brainpy_state` Package — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the flat `brainpy_state/_nest/` package into five category packages (`_nest_base`, `_nest_neuron`, `_nest_synapse`, `_nest_plasticity`, `_nest_device`), rename `_network/`→`_nest_network/` and `_nest/_validation/`→`_nest_validation/`, wire 21 synapse/plasticity spec classes to their NEST marker base classes, and relocate internal dev docs — all with the public API (`brainpy_state.*`) byte-for-byte unchanged.

**Architecture:** Mostly mechanical move + import-rewrite, driven by one deterministic migration script whose categorization function already classified all 248 files with zero unclassified. The synapse↔network import cycle is preserved by keeping `_nest_network/__init__.py`'s load order. Marker wiring mirrors the existing `gap_junction(NESTSynapse)` precedent (`super().__init__(in_size=1)`).

**Tech Stack:** Python, `git mv` (history preservation), `pytest`, `brainstate`/`brainunit`/`jax`.

**Spec:** `2026-06-16-restructure-package-spec.md` (read it first).

---

## File Structure

New packages under `../../brainpy_state` (each a directory with `__init__.py`):

- `_nest_base/` — `_base.py`, `_utils.py`, `_plastic_base.py` (+ `_utils_test.py`, `jit_compat_test.py`)
- `_nest_neuron/` — 67 neuron model stems + tests
- `_nest_synapse/` — static synapses + connections + STP (13 stems) + tests
- `_nest_plasticity/` — STDP + voltage-based (15 stems) + tests (+ `stdp_docs_crosslink_test.py`)
- `_nest_device/` — generators, recorders, detectors (27 stems) + tests
- `_nest_network/` ← `_network/` (whole tree)
- `_nest_validation/` ← `_nest/_validation/` (whole tree)

Hand-edited files: `../../brainpy_state/__init__.py` (import block), the five new `__init__.py` (script-generated), `_base.py`→`_nest_base/_base.py` (docstrings), 21 spec files (marker wiring), `docs/api/nest-base.rst` (ASCII annotation), `changelog.md` + moved docs (textual path refs).

The migration script lives at `` (gitignored; not part of the repo history).

---

## Task 1: Worktree + baseline snapshot

**Files:** none modified (setup + read-only capture).

- [ ] **Step 1: Create the isolated worktree**

Use the **superpowers:using-git-worktrees** skill to create a worktree branch (CLAUDE.md rule 6 — work must not happen on `main`). Suggested branch: `refactor/nest-package-split`.

- [ ] **Step 2: Make the gitignored spec + plan available in the worktree**

`../../dev` is gitignored, so the spec/plan don't travel with the worktree. Copy them in (adjust source path to the original checkout):

```bash
mkdir -p dev/superpowers
cp /mnt/d/codes/projects/brainpy-state/dev/superpowers/restructure-package-spec.md dev/superpowers/
cp /mnt/d/codes/projects/brainpy-state/dev/superpowers/restructure-package-plan.md dev/superpowers/
```

- [ ] **Step 3: Capture the baseline (run from worktree root, BEFORE any change)**

```bash
python -c "import brainpy_state" && echo "IMPORT_OK"
python -c "import brainpy_state as b; print('\n'.join(sorted(b.__all__)))" > dev/superpowers/all_before.txt
wc -l dev/superpowers/all_before.txt
python -m pytest brainpy_state/ -m "not requires_nest" --co -q 2>/dev/null | tail -3 | tee dev/superpowers/collect_before.txt
```

Expected: prints `IMPORT_OK`; `all_before.txt` has the full public surface; collection finishes with a test count and **no import errors**. Record the collected-count number.

---

## Task 2: Wire spec classes to NEST marker base classes (pre-move)

Do this **before** the move so it is an isolated, reviewable behavioral change against the known-green tree. Files are still at `brainpy_state/_nest/<model>.py`; the later import-rewrite converts the `from ._base import …` added here to `from brainpy_state._nest_base._base import …`.

**Files (21 specs + base docstrings):**
- Modify (add `NESTSynapse` base): `_nest/static_synapse.py`, `static_synapse_hom_w.py`, `bernoulli_synapse.py`, `cont_delay_synapse.py`, `tsodyks_synapse.py`, `tsodyks_synapse_hom.py`, `tsodyks2_synapse.py`, `quantal_stp_synapse.py`
- Modify (add `NESTPlasticity` base): `_nest/stdp_synapse.py`, `stdp_triplet_synapse.py`, `stdp_dopamine_synapse.py`, `stdp_facetshw_synapse_hom.py`, `stdp_nn_pre_centered_synapse.py`, `stdp_nn_restr_synapse.py`, `stdp_nn_symm_synapse.py`, `stdp_pl_synapse_hom.py`, `clopath_synapse.py`, `urbanczik_synapse.py`, `vogels_sprekeler_synapse.py`, `ht_synapse.py`, `jonke_synapse.py`
- Verify only (already inherits via parent): `_nest/stdp_synapse_hom.py`
- Modify (docstrings): `_nest/_base.py`

- [ ] **Step 1: Recipe per spec file** (apply to each file in the two lists above)

For a `NESTSynapse` spec (worked example — `tsodyks_synapse.py`):

1. Add the marker import next to the existing `from ._plastic_base import …`:
   ```python
   from ._base import NESTSynapse
   ```
2. Change the class header:
   ```python
   class tsodyks_synapse(NESTSynapse):
   ```
3. Insert `super().__init__()` as the **first statement** of `__init__` (immediately after the `):` that closes the signature), at the method-body indent:
   ```python
       def __init__(self, weight: ArrayLike = 1.0, delay=..., ...):
           super().__init__(in_size=1)
           # ... existing body unchanged ...
   ```

For a `NESTPlasticity` spec, use `from ._base import NESTPlasticity` and `class <name>(NESTPlasticity):` instead. Everything else (the `update` kernel, parameters, existing `__module__ = 'brainpy.state'`) is unchanged.

**`stdp_synapse_hom.py`:** do NOT add a base (it is `class stdp_synapse_hom(stdp_synapse):`). Instead confirm its `__init__` calls `super().__init__(...)` so the chain reaches `Dynamics`. If it does not call super, add `super().__init__(in_size=1)` only if it does not already delegate to `stdp_synapse.__init__` (which now calls it). Read the file and choose the minimal correct edit.

- [ ] **Step 2: Update `_nest/_base.py` docstrings**

In the `NESTSynapse` docstring, extend the enumeration to include short-term plasticity, e.g.:
```
Covers static synapses (``static_synapse``, ``static_synapse_hom_w``,
``bernoulli_synapse``, ``cont_delay_synapse``), short-term-plasticity synapses
(``tsodyks_synapse``, ``tsodyks_synapse_hom``, ``tsodyks2_synapse``,
``quantal_stp_synapse``), gap junctions (``gap_junction``,
``diffusion_connection``), rate connections (``rate_connection_instantaneous``,
``rate_connection_delayed``), and other non-plastic connection models
(``sic_connection``).
```
In the `NESTPlasticity` docstring, remove the **Short-Term Plasticity (STP)** family block, leaving only STDP and Voltage-Based / Specialised; update the lead sentence to "marks long-term / activity-history plasticity synapse models (STDP and voltage-based learning rules)."

- [ ] **Step 3: Verify import + marker membership + plastic substrate**

```bash
python -c "import brainpy_state" && echo IMPORT_OK
python -c "
import brainpy_state as b
assert isinstance(b.tsodyks_synapse(), b.NESTSynapse), 'tsodyks not NESTSynapse'
assert isinstance(b.quantal_stp_synapse(), b.NESTSynapse), 'quantal not NESTSynapse'
assert isinstance(b.stdp_synapse(), b.NESTPlasticity), 'stdp not NESTPlasticity'
assert isinstance(b.clopath_synapse(), b.NESTPlasticity), 'clopath not NESTPlasticity'
assert isinstance(b.stdp_synapse_hom(), b.NESTPlasticity), 'stdp_hom not NESTPlasticity'
assert not isinstance(b.tsodyks_synapse(), b.NESTPlasticity), 'tsodyks wrongly NESTPlasticity'
print('MARKERS_OK')
"
python -m pytest brainpy_state/_network/_event_plastic_test.py \
  brainpy_state/_network/_simulator_plastic_test.py \
  brainpy_state/_network/_simulator_dopamine_test.py \
  brainpy_state/_network/_voltage_coupled_plastic_test.py \
  -m "not requires_nest" -q
```

Expected: `IMPORT_OK`, `MARKERS_OK`, all four substrate test files pass. If a spec's `__init__` requires args (no defaults), instantiate it in the check with minimal valid args instead of `()`.

- [ ] **Step 4: Commit**

```bash
git add brainpy_state/_nest/
git commit -m "refactor(nest): wire synapse/plasticity specs to NESTSynapse/NESTPlasticity markers"
```

---

## Task 3: Migration script — split, rename, rewrite imports (atomic)

The move and the import-rewrite must land in one commit so the tree is never half-broken.

**Files:** Create ``; it moves all `_nest/*.py`, `_nest/_validation`, `_network`, generates 5 `__init__.py`, rewrites imports across the package, and patches `../../brainpy_state/__init__.py`.

- [ ] **Step 1: Write ``**

```python
#!/usr/bin/env python3
"""One-shot migration. Run from the worktree repo root: python dev/superpowers/migrate.py"""
import os, re, subprocess

PKG, NEST = "brainpy_state", "brainpy_state/_nest"
CATS = ["_nest_base", "_nest_neuron", "_nest_synapse", "_nest_plasticity", "_nest_device"]

def stem(fname):
    s = fname[:-3]
    for suf in ("_rule_test", "_state_test", "_test"):
        if s.endswith(suf):
            return s[:-len(suf)]
    return s

def categorize(fname):
    """Proven over all 248 files: 0 unclassified."""
    s = stem(fname)
    if fname in ("jit_compat_test.py", "_utils_test.py"): return "_nest_base"
    if s in ("_base", "_utils", "_plastic_base"): return "_nest_base"
    if fname == "stdp_docs_crosslink_test.py": return "_nest_plasticity"
    if re.match(r"^(iaf_|aeif_|gif_|glif_|hh_|amat2_|mat2_|pp_)", s): return "_nest_neuron"
    if s in {"ht_neuron","izhikevich","erfc_neuron","ginzburg_neuron","mcculloch_pitts_neuron",
             "astrocyte_lr_1994","ignore_and_fire","parrot_neuron","cm_default",
             "rate_neuron_ipn","rate_neuron_opn","rate_transformer_node","siegert_neuron",
             "gauss_rate","lin_rate","sigmoid_rate","sigmoid_rate_gg_1998","tanh_rate",
             "threshold_lin_rate"}:
        return "_nest_neuron"
    if s in {"static_synapse","static_synapse_hom_w","bernoulli_synapse","cont_delay_synapse",
             "gap_junction","diffusion_connection","sic_connection",
             "rate_connection_delayed","rate_connection_instantaneous",
             "tsodyks_synapse","tsodyks_synapse_hom","tsodyks2_synapse","quantal_stp_synapse"}:
        return "_nest_synapse"
    if re.match(r"^stdp_", s): return "_nest_plasticity"
    if s in {"clopath_synapse","urbanczik_synapse","vogels_sprekeler_synapse","ht_synapse","jonke_synapse"}:
        return "_nest_plasticity"
    if re.search(r"generator|detector|drive", s): return "_nest_device"
    if s in {"multimeter","voltmeter","spike_recorder","spin_detector","weight_recorder",
             "volume_transmitter","spike_train_injector","spike_dilutor"}:
        return "_nest_device"
    raise SystemExit(f"UNCLASSIFIED: {fname}")

def git(*a): subprocess.run(["git", *a], check=True)

nest_files = sorted(f for f in os.listdir(NEST) if f.endswith(".py") and f != "__init__.py")

# module(stem of source file) -> package, for import rewriting
MOD2PKG = {f[:-3]: categorize(f) for f in nest_files if not f.endswith("_test.py")}
MOD2PKG.update({"_base": "_nest_base", "_utils": "_nest_base", "_plastic_base": "_nest_base"})

# capture the original aggregator BEFORE deleting it
orig = open(os.path.join(NEST, "__init__.py")).read()
import_lines = re.findall(r"^from \.(\w+) import (.+)$", orig, re.M)
all_block = re.search(r"__all__ = \[(.*?)\]", orig, re.S).group(1)
all_names = re.findall(r"'([^']+)'", all_block)

cat_imports = {c: [] for c in CATS}
name2cat = {}
for mod, names in import_lines:
    c = categorize(mod + ".py")
    cat_imports[c].append(f"from .{mod} import {names}")
    for nm in (x.strip() for x in names.split(",")):
        name2cat[nm] = c
cat_all = {c: [] for c in CATS}
for nm in all_names:
    cat_all[name2cat[nm]].append(nm)

# 1) create package dirs and move files
for c in CATS:
    os.makedirs(os.path.join(PKG, c), exist_ok=True)
for f in nest_files:
    git("mv", os.path.join(NEST, f), os.path.join(PKG, categorize(f), f))

# 2) write the five __init__.py (partitioned from the original)
HEADER = "# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.\n"
for c in CATS:
    body = [HEADER, ""] + cat_imports[c] + ["", "__all__ = ["] \
           + [f"    '{nm}'," for nm in cat_all[c]] + ["]", ""]
    open(os.path.join(PKG, c, "__init__.py"), "w").write("\n".join(body))

# 3) rename whole-tree packages, drop old _nest
git("mv", os.path.join(NEST, "_validation"), os.path.join(PKG, "_nest_validation"))
git("mv", os.path.join(PKG, "_network"), os.path.join(PKG, "_nest_network"))
git("rm", "-q", os.path.join(NEST, "__init__.py"))
os.rmdir(NEST)

UNMAPPED = set()
def rewrite_absolute(t):
    t = t.replace("brainpy_state._nest._validation", "brainpy_state._nest_validation")
    t = t.replace("brainpy_state._network", "brainpy_state._nest_network")
    def repl(m):
        mod = m.group(1); pkg = MOD2PKG.get(mod)
        if pkg is None:
            UNMAPPED.add(mod); return m.group(0)
        return f"brainpy_state.{pkg}.{mod}"
    return re.sub(r"brainpy_state\._nest\.(\w+)", repl, t)

def rewrite_relative(t, this_cat):
    def repl(m):
        mod = m.group(1); pkg = MOD2PKG.get(mod)
        if pkg is None or pkg == this_cat:
            return m.group(0)
        return f"from brainpy_state.{pkg}.{mod} import"
    return re.sub(r"from \.(\w+) import", repl, t)

# 4) Pass A (relative) on the 5 category packages, then Pass B (absolute) everywhere
for c in CATS:
    d = os.path.join(PKG, c)
    for f in os.listdir(d):
        if f.endswith(".py"):
            p = os.path.join(d, f)
            open(p, "w").write(rewrite_relative(open(p).read(), c))
for root, _, files in os.walk(PKG):
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(root, f)
            open(p, "w").write(rewrite_absolute(open(p).read()))

# 5) patch brainpy_state/__init__.py (relative forms Pass B does not touch)
ip = os.path.join(PKG, "__init__.py"); t = open(ip).read()
blocks = []
for c in CATS:
    blocks.append(f"from .{c} import (")
    blocks += [f"    {nm}," for nm in cat_all[c]]
    blocks.append(")")
t = re.sub(r"from \._nest import \([^)]*\)", "\n".join(blocks), t, flags=re.S)
t = t.replace("from ._network import", "from ._nest_network import")
t = t.replace("from . import _network as network", "from . import _nest_network as network")
open(ip, "w").write(t)

git("add", "-A", PKG)
print("UNMAPPED (should be empty):", sorted(UNMAPPED))
print("done")
```

- [ ] **Step 2: Run the migration**

```bash
python dev/superpowers/migrate.py
```

Expected: ends with `UNMAPPED (should be empty): []` and `done`. If `UNMAPPED` is non-empty, stop — a `brainpy_state._nest.<x>` reference points at an unknown module; investigate before continuing.

- [ ] **Step 3: Verify importability and the public surface**

```bash
python -c "import brainpy_state" && echo IMPORT_OK
python -c "import brainpy_state as b; print('\n'.join(sorted(b.__all__)))" > dev/superpowers/all_after.txt
diff dev/superpowers/all_before.txt dev/superpowers/all_after.txt && echo ALL_IDENTICAL
```

Expected: `IMPORT_OK`, then `ALL_IDENTICAL` (zero diff — public API unchanged). If `import` raises a circular-import error, confirm `_nest_network/__init__.py` still imports `_event_plastic` before `_simulator` (it should, untouched).

- [ ] **Step 4: Verify no stale paths remain**

```bash
git grep -nE "brainpy_state\._nest\b" -- '*.py' || echo "NONE_nest"
git grep -nE "brainpy_state\._network\b" -- '*.py' || echo "NONE_network"
git grep -nE "from \._nest import|from \._network import|import _network as" -- 'brainpy_state/__init__.py' || echo "NONE_init"
```

Expected: `NONE_nest`, `NONE_network`, `NONE_init` (each grep returns nothing → the `|| echo` fires). Any hits are stale references — fix them and re-run.

- [ ] **Step 5: Verify test collection parity**

```bash
python -m pytest brainpy_state/ -m "not requires_nest" --co -q 2>/dev/null | tail -3
```

Expected: same collected-count as the Task 1 baseline, **no import/collection errors**. (Test files moved with their modules, so the count is unchanged.)

- [ ] **Step 6: Confirm history preservation on a sample**

```bash
git log --follow --oneline -- brainpy_state/_nest_neuron/iaf_psc_alpha.py | head -3
git log --follow --oneline -- brainpy_state/_nest_network/_simulator.py | head -3
```

Expected: each shows pre-move history (the `git mv` preserved it).

- [ ] **Step 7: Commit**

```bash
git commit -m "refactor(nest): split _nest into _nest_base/_neuron/_synapse/_plasticity/_device; rename _network->_nest_network, _validation->_nest_validation"
```

---

## Task 4: Run the full (non-NEST) test suite

**Files:** none (verification gate before the doc moves).

- [ ] **Step 1: Full suite**

```bash
python -m pytest brainpy_state/ -m "not requires_nest" -q -x 2>&1 | tail -30
```

Expected: all selected tests pass. If a test fails on an import path the script missed, fix the offending import, re-run, and amend Task 3's commit (`git commit --amend --no-edit` after `git add`).

---

## Task 5: Move internal docs & examples; update textual references

**Files:**
- Move: `examples/nest/`→`../../examples/nest_like`; `CONTEXT.md`→`develop/NEST_PARITY_LEDGER.md`; `DESIGN_delays_and_missing_features.md`→`develop/`; `docs/nest-status/internal/`→`develop/nest-status/internal/`
- Modify (textual refs): moved `.py` docstrings/comments, `../../changelog.md`, `develop/NEST_PARITY_LEDGER.md`, `docs/api/nest-base.rst`

- [ ] **Step 1: Move with `git mv`**

```bash
git mv examples/nest examples/nest_like
git mv CONTEXT.md develop/NEST_PARITY_LEDGER.md
git mv DESIGN_delays_and_missing_features.md develop/DESIGN_delays_and_missing_features.md
mkdir -p develop/nest-status
git mv docs/nest-status/internal develop/nest-status/internal
```

- [ ] **Step 2: Update textual references**

```bash
python - <<'PY'
import re, subprocess
files = subprocess.check_output(
    ["git", "grep", "-lE",
     r"examples/nest/|docs/nest-status/internal|CONTEXT\.md|DESIGN_delays_and_missing_features",
     "--", "*.py", "*.md", "*.rst"]).decode().split()
for p in files:
    t = open(p).read(); o = t
    t = t.replace("examples/nest/", "examples/nest_like/")
    t = t.replace("docs/nest-status/internal", "develop/nest-status/internal")
    t = t.replace("CONTEXT.md", "develop/NEST_PARITY_LEDGER.md")
    t = t.replace("DESIGN_delays_and_missing_features.md",
                  "develop/DESIGN_delays_and_missing_features.md")
    # avoid double-prefixing if a path was already under develop/
    t = t.replace("develop/develop/", "develop/")
    if t != o:
        open(p, "w").write(t); print("updated", p)
PY
```

- [ ] **Step 3: Fix the ASCII annotation in `docs/api/nest-base.rst`**

Replace the four `(_nest/_base.py)` annotations in the class-tree diagram with `(_nest_base/_base.py)` (use Edit; they are inside a literal code block, so the path text is cosmetic only).

- [ ] **Step 4: Verify the crosslink test and a clean grep**

```bash
python -m pytest brainpy_state/_nest_plasticity/stdp_docs_crosslink_test.py -m "not requires_nest" -q
git grep -nE "examples/nest/|docs/nest-status/internal|(^|[^/])CONTEXT\.md" -- '*.py' '*.md' '*.rst' || echo CLEAN
```

Expected: the crosslink test passes (it points at `docs/nest-guide/stdp-divergences.rst`, which did not move); `CLEAN` (no stale path refs remain, modulo intended ones).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs: move internal dev docs into develop/, rename examples/nest -> examples/nest_like, CONTEXT.md -> NEST_PARITY_LEDGER.md"
```

---

## Task 6: Final verification & packaging sanity

**Files:** none.

- [ ] **Step 1: Re-import, re-collect, packaging check**

```bash
python -c "import brainpy_state; print('OK', len(brainpy_state.__all__))"
python -m pytest brainpy_state/ -m "not requires_nest" --co -q 2>/dev/null | tail -1
python -c "from setuptools import find_packages; ps=find_packages(include=['brainpy_state*']); print([p for p in ps if p.startswith('brainpy_state._nest')])"
```

Expected: import OK; collected-count matches the Task 1 baseline; `find_packages` lists `brainpy_state._nest_base/_neuron/_synapse/_plasticity/_device/_nest_network/_nest_validation` (the `brainpy_state*` wildcard in `../../pyproject.toml` auto-discovers them — no packaging edit needed).

- [ ] **Step 2: Optional — run the live-NEST parity suite if NEST is installed**

```bash
python -c "import nest" 2>/dev/null && python -m pytest brainpy_state/_nest_validation/ -m requires_nest -q || echo "NEST not installed; skipping parity suite"
```

- [ ] **Step 3: Finish the branch**

Use the **superpowers:finishing-a-development-branch** skill to choose merge / PR / cleanup. The `` script is gitignored and need not be committed.

---

## Notes for the executor

- **Commit discipline:** four commits (marker wiring → structural move → full-suite gate is part of the move commit if a fix is needed → doc moves). Keep each green.
- **If the migration script misses an import form:** the Task 3 Step 4 greps catch bare `_nest`/`_network` refs; the Task 4 suite catches runtime breakage. Fix forward and amend the move commit.
- **Do not** edit any model's `update`/rule kernel or NEST parameters — only the marker base + `super().__init__` prologue (Task 2) and import paths (Task 3) change.
- **Circular import:** if `import brainpy_state` fails after the move, the cause is almost certainly load-order in `_nest_network/__init__.py` — it must import `_event_plastic` before `_simulator`. The script does not touch that file's order.
