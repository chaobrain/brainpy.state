# Spec — `brainpy` ↔ `brainpy.state` relationship declaration

**Goal.** Add a Numba-style declaration that states the relationship between classic
`brainpy` and `brainpy.state`, cross-referenced from **both** doc sites.

**Model.** Numba's "CUDA Deprecation Status" notice — its *two-part structure* (short
titled admonition ➜ dedicated detailed block) and *neutral, reassuring tone*.
**Not** its deprecation semantics (see D2).

---

## Established facts (from code, not assumptions)

- `brainpy.state` is delivered by the standalone **`brainpy_state`** distribution and
  surfaced through the `brainpy.state` namespace via a shim: `brainpy/state/__init__.py`
  = `from brainpy_state import *`; classic `brainpy/__init__.py:146` does
  `from brainpy import state`.
- **Mutual dependency** → both install paths land in the same place:
  - classic `brainpy` depends on `brainpy_state>=0.0.3` (first shipped in **V2.7.5**;
    current `brainpy` = 2.8.1), so `pip install brainpy` already bundles the layer;
  - `brainpy_state` depends on `brainpy>=2.7.6`, so `pip install brainpy.state` pulls
    `brainpy` in too.
  - ⇒ honest floor to cite: **"bundled with `brainpy >= 2.7.6`"**; "no code changes
    required" is true.
- Existing (informal) relationship prose already lives in `brainpy/state/README.md`
  ("maintained as a separate package `brainpy_state` and re-exported through BrainPy").
  We are **formalizing/upgrading** this, not inventing it. That README also contains a
  contradictory install claim to fix.
- Classic `brainpy` modeling surface = `DynamicalSystem`-based: `brainpy.dyn`,
  `brainpy.math`, `brainpy.integrators` (+ `dynold`, `neurons`, `synapses`, `channels`).
- This repo's docs: Sphinx (`sphinx_book_theme`, `sphinx_design`, `myst_nb`,
  `sphinx.ext.intersphinx`). `intersphinx_mapping` = python + sphinx **only** (no brainpy
  mapping). House callout style = custom-titled `.. admonition::` (16×) over `.. note::`
  (6×).
- Canonical brainpy.state docs URL (per `../../pyproject.toml` + README badge) =
  `https://brainx.chaobrain.com/brainpy-state/`. Classic brainpy docs =
  `https://brainpy.readthedocs.io/`.

---

## Decisions (from grilling)

- **D1 — Relationship semantics = (C):** `brainpy.state` is separately maintained **and**
  the recommended forward direction for new state-based work; classic `brainpy` remains
  fully supported.
- **D2 — Framing = (C-additive):** **no** deprecation/sunset language. Borrow Numba's
  *structure only*. `brainpy.state` = a new, separately-released layer; classic `brainpy`
  unchanged and fully supported; the two paradigms coexist.
- **D3 — Topology = (iii) hybrid:** one canonical detailed declaration lives in the
  **brainpy.state** docs; short admonitions in both landings (and the brainpy-side
  README) point to it.
- **D4 — Placement = (a):** canonical detailed block is a **dedicated page**
  `../project/relationship.rst`, added to the `Project` toctree. Two-part shape
  confirmed (landing admonition + dedicated page).
- **D5 — Terminology:** "classic `brainpy`" (never "legacy"/version-numbered); name both
  paradigms (`DynamicalSystem`-based `brainpy.dyn`/`brainpy.math`/`brainpy.integrators`
  vs `brainstate.State`-based `brainpy.state`); prose says `brainpy.state`, mention
  `brainpy_state` exactly once (the release mechanic).
- **D6 — Versions = (b):** evergreen + one floor note "bundled with `brainpy >= 2.7.6`".
  No removal timeline.
- **D7 — Page scope = (b):** minimal / Numba-parity **plus** a short "which paradigm
  when" note. No interop/migration deep-dive.
- **D8 — RST specifics:** custom-titled `.. admonition:: brainpy and brainpy.state`;
  landing admonition placed **after the two opening paragraphs, before the bridge
  figure**; dedicated page added to `Project` toctree (after `changelog`, by
  `ecosystem`/`citing`); install via plain `.. code-block:: bash`; internal links via
  `:doc:`, cross-site links via plain external URLs.
- **D9 — Cross-repo scope (recommended, needs your OK):** implement the brainpy.state
  side fully in **this** repo on a worktree branch (rule 6). Deliver the classic-`brainpy`
  side (a matching admonition in their docs + the `brainpy/state/README.md` install fix)
  as a ready-to-apply snippet with exact targets; apply it in the separate
  `gitee brainpy/BrainPy` repo only on your confirmation (different repo/remote/PR). Exact
  insertion point in the brainpy docs to be confirmed during implementation.

---

## Deliverables

### A. `../index.rst` — landing admonition (this repo)
Insert after the two opening paragraphs, before `.. figure:: /_static/bridging-concept.png`:

```rst
.. admonition:: ``brainpy`` and ``brainpy.state``

   ``brainpy.state`` is the **state-based modeling layer** of BrainPy — developed and
   released as the standalone ``brainpy_state`` package and surfaced through the
   ``brainpy.state`` namespace. It is the recommended starting point for new
   ``State``-based, differentiable spiking-network models. The classic,
   ``DynamicalSystem``-based ``brainpy`` API (``brainpy.dyn``, ``brainpy.math``,
   ``brainpy.integrators``) is unchanged and fully supported — the two paradigms
   coexist. See :doc:`project/relationship`.
```

### B. `../project/relationship.rst` — canonical page (this repo)
```rst
Relationship between ``brainpy`` and ``brainpy.state``
======================================================

``brainpy.state`` is the state-based modeling layer of `BrainPy
<https://brainpy.readthedocs.io/>`_. It is developed and released as the standalone
``brainpy_state`` package and surfaced through the ``brainpy.state`` namespace, so the
same code is reachable as ``brainpy.state`` whether you install classic ``brainpy`` or
the layer on its own.

Why a separate package
----------------------

- **Decoupled release cycle.** ``brainpy.state`` is versioned and released independently
  of classic ``brainpy``, so its models, fixes, and features ship on their own cadence.
- **Surfaced through the ``brainpy.state`` namespace.** The standalone ``brainpy_state``
  distribution is re-exported by classic ``brainpy`` as ``brainpy.state`` — no separate
  import path to learn.
- **Built on the BrainX substrate.** Built on ``brainstate`` (State management),
  ``brainunit`` (physical units), and the rest of the `BrainX ecosystem
  <https://brainx.chaobrain.com/>`_, compiling through JAX to CPU, GPU, and TPU.

Which paradigm to use
---------------------

- Reach for **``brainpy.state``** for new work: ``State``-based models, surrogate-gradient
  / differentiable training, online learning, and JAX-native pipelines. This is the
  recommended starting point.
- Keep using **classic ``brainpy``** for existing ``DynamicalSystem`` models and for the
  array/operator and integrator API (``brainpy.math``, ``brainpy.integrators``,
  ``brainpy.dyn``). It is unchanged and fully supported.

The two paradigms coexist; adopting ``brainpy.state`` does not deprecate or remove any
part of classic ``brainpy``.

Installation
------------

``brainpy.state`` is bundled with classic ``brainpy`` (``brainpy >= 2.7.6``), so if you
installed ``brainpy`` you already have it — no code changes required:

.. code-block:: python

   import brainpy
   neuron = brainpy.state.LIF(...)

To install or upgrade the layer on its own release cycle:

.. code-block:: bash

   pip install -U brainpy.state

See Also
--------

- :doc:`/project/ecosystem` — the wider BrainX ecosystem ``brainpy.state`` builds on.
- `Classic BrainPy documentation <https://brainpy.readthedocs.io/>`_ — the
  ``DynamicalSystem``-based API.
```

### C. `../index.rst` — Project toctree
Add `project/relationship` after `../../changelog.md` (before/around `project/citing`).

### D. Classic-`brainpy` side (separate repo — snippet, apply on confirmation)
- Titled admonition for the brainpy docs landing/quickstart (exact page TBD):

```rst
.. admonition:: ``brainpy`` and ``brainpy.state``

   ``brainpy.state`` is the state-based modeling layer of BrainPy — developed and
   released as the standalone ``brainpy_state`` package and surfaced here through the
   ``brainpy.state`` namespace (bundled with ``brainpy >= 2.7.6``; no separate install
   needed). It is the recommended starting point for new ``State``-based, differentiable
   spiking-network models, while the classic ``DynamicalSystem``-based API is unchanged
   and fully supported. See the `brainpy.state relationship page
   <https://brainx.chaobrain.com/brainpy-state/project/relationship.html>`_.
```

- Fix `brainpy/state/README.md` install section: state that `brainpy.state` ships with
  `brainpy` **and** can be installed/upgraded standalone via `pip install -U brainpy.state`
  (remove the either/or contradiction).

---

## Out of scope
- No API/code changes; docs only.
- No migration guide, no interop deep-dive, no version-support table.

## Validation
- `..` Sphinx build succeeds; no new broken-xref warnings for `project/relationship`,
  the `:doc:` links, or the landing admonition.
- Rendered admonition + page read correctly (titled callout; toctree entry present).
- Wording contains **no** deprecation/sunset language (D2); floor note reads
  `brainpy >= 2.7.6` (D6).

## Execution notes
- Rule 6: do this on a **worktree branch**, not `main`.
- Rule 1: await approval of this spec before editing.
