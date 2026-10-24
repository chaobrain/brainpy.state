# brainspike

## Working agreement

1. Before writing any code, describe approach, wait for approval.
2. Requirements ambiguous? Ask clarifying questions before writing code.
3. After writing code, list edge cases + suggest test cases.
4. Bug? Write a test that reproduces it, then fix until the test passes.
5. Every correction: reflect on the mistake, plan to avoid repeating it.
6. All updates must be happened on the worktree branch, not main. 
7. Write a spec only when it will actually guide the implementation: the work
   spans several modules or repos, changes a public API or the architecture,
   has real design choices to settle first, or will take multiple sessions.
   Skip it for bug fixes, small features, local refactors, and docs edits.
   A spec goes in `docs/specs/YYYY-MM-DD-<slug>.md`, one file (no separate
   -spec/-plan pairs), and stays short: goal, decisions with reasons, step
   plan, how to verify. Specs are dated records: don't rewrite old ones to
   match later changes.
8. Tests should >90% coverage, but focus on meaningful tests that cover edge cases and critical paths, not just trivial lines. 
9. **One namespace: `brainspike`.**
    - **Example code (docstrings, `examples/`, docs)** → use the public API (`import brainspike`, or `from brainspike import ...`); examples must run against the surface end users actually import, and must NOT import private `brainspike._*` submodules (inline or re-derive what they need from the public API instead).
    - **Inner package source and tests** → import `brainspike` / `brainspike._*` directly.
    - **`brainpy` is a dev-only dependency.** Shipped code never imports `brainpy`. The one exception is the kernel parity tests (`brainspike/_kernel/*_test.py`), which import BrainPy-style reference models from `brainpy.state` (listed in `requirements-dev.txt`).
10. **Never drive a model with a bare Python `for`/`while` loop when it runs repeatedly.** Python loops execute op-by-op (dispatch overhead, no fusion) and trace fresh each step; the `brainstate.transform` primitives lower the whole loop into one compiled XLA program, tracing the body only once. Pick by shape of the work:
    - **Single step or one-shot call** → `brainstate.transform.jit` — wrap the step/model call so it compiles once and reuses the trace.
    - **Many steps, collect outputs** → `brainstate.transform.for_loop` — repeat a step `length` times or map over `xs`; `State` is carried automatically and stacked outputs are returned.
    - **Many steps with an explicit carry** → `brainstate.transform.scan` — when threading a carry value alongside `State` (`f(carry, x) -> (carry, y)`).
    - **Long rollout under autograd (backprop through time)** → `brainstate.transform.checkpointed_for_loop` / `brainstate.transform.checkpointed_scan` — same semantics as above but rematerialize activations on the backward pass (tune `base`) to bound peak memory at the cost of recomputation.

    Compose them freely (e.g. `jit` an outer driver that calls a `for_loop`/`scan`). Reach for the checkpointed variants only when reverse-mode gradients through a long simulation would otherwise exhaust memory — otherwise prefer plain `for_loop`/`scan`.


## Docstring style (NumPy-doc)

All public classes, methods, functions must use [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html). Rules for the Examples section:

- Wrap example code in `.. code-block:: python` directive so Sphinx render with syntax highlighting.
- Prefix every input line with `>>>` (continuation lines with `...`) for `doctest` compatibility.
- Show expected output on line immediately after statement, **without** prompt prefix.
- Separate distinct scenarios with blank `>>>` line.
- Always include necessary imports (`import brainunit as u`, etc.) at top of example block so self-contained.

