# brainpy_state

## Working agreement

1. Before writing any code, describe approach, wait for approval.
2. Requirements ambiguous? Ask clarifying questions before writing code.
3. After writing code, list edge cases + suggest test cases.
4. Bug? Write a test that reproduces it, then fix until the test passes.
5. Every correction: reflect on the mistake, plan to avoid repeating it.
6. All updates must be happened on the worktree branch, not main. 
7. Write spec under `docs/specs` before implementation. 
8. Tests should >90% coverage, but focus on meaningful tests that cover edge cases and critical paths, not just trivial lines. 
9. **Two namespaces, two contexts — never cross them, or development hits circular imports.**
    - **Example code (docstrings, `examples/`, docs)** → use the public `brainpy.state` namespace (`from brainpy import state as ...`, or `from brainpy.state import ...`); examples must run against the surface end users actually import, and must NOT import private `brainpy_state._*` submodules (inline or re-derive what they need from the public API instead).
    - **Inner package source and tests** → import `brainpy_state` directly (`from brainpy_state import ...`). Reaching the package through `brainpy.state` from inside it re-enters `brainpy`'s import machinery while `brainpy_state` is still initializing — a partially-initialized-module / circular-import error — and can resolve to an installed `brainpy` instead of the local source under test.
    - **Enforced.** `brainpy_state/_namespace.py::enforce_namespace_access()` runs at package-init top and raises `ImportError` on a direct external `import brainpy_state`, pointing users at `brainpy.state`. The blessed `brainpy.state` shim, `pytest`, `sphinx`, and the `BRAINPY_STATE_ALLOW_DIRECT_IMPORT=1` override are exempt — so internal source/tests/doc-builds keep importing `brainpy_state` directly.
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

