# Spec & Plan — Fix `_check_brainpy_compatibility()`

Status: approved via grill-me interview (2026-06-16). Gitignored reference doc.

## Problem

`../../brainpy_state/__init__.py` defines `_check_brainpy_compatibility()` whose
`raise RuntimeError(...)` lives *inside* a `try` ended by a bare `except: pass`.
The function catches the very exception it raises — **the check is a complete
no-op**; it can never block an import.

Secondary defects:
- Threshold mismatch: `../../pyproject.toml` requires `brainpy>=2.7.6` but the check
  used `>= 2.7.5`.
- Message advertises "or no brainpy installed" as a normal state, contradicting
  the hard `brainpy>=2.7.6` dependency.
- Fragile chained `.split(...)` parsing (e.g. `"2.7"` would `IndexError`).
- `PackageNotFoundError` imported but unused.
- No test coverage; function `del`-eted after call, so untestable as written.

## Decisions (from interview)

1. **Contract (B):** brainpy is genuinely required; the check is a *defensive
   guard* against a stale/downgraded brainpy (e.g. `--no-deps`, pinned lockfiles).
   "brainpy absent" is tolerated silently (rare `--no-deps` case), not advertised.
2. **Failure mode (A):** hard fail, raised as **`ImportError`** (unsatisfiable
   import condition), not `RuntimeError`, not a soft `warn`.
3. **Threshold source-of-truth (C):** hardcode literal `MIN_BRAINPY = (2, 7, 6)`;
   add a test that parses `../../pyproject.toml` and asserts it matches — drift caught
   in CI, runtime path stays a dead-simple literal.
4. **Parsing (B):** dependency-free regex on the PEP 440 release segment:
   `re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", v)` → `(major, minor, patch)`.
   No `packaging` dependency (a robustness guard must not add an import that
   could itself be absent). Present-but-unparseable → `warn` + allow.
5. **Structure (raise-outside / no safety net):** compute inside `try`, **raise
   outside**. Only `except PackageNotFoundError`. No catch-all — genuine internal
   bugs propagate loudly. Original bug becomes structurally impossible.
6. **Extraction:** move to `../../brainpy_state/_compat.py` with a pure
   `_parse_release` helper + injectable `min_version`, solving the
   deleted-function testability problem and decluttering `__init__.py`.
7. **Message:** single actionable instruction (upgrade); threshold interpolated
   from `min_version` so no fourth copy drifts; name `brainpy.state` in warnings.
8. **Semantics:** `2.7.6.dev0` / `2.7.6rc1` of the *target* version are allowed
   (release-segment compare) — don't block devs on pre-releases of the threshold.

## Implementation

### `../../brainpy_state/_compat.py` (new)
```python
import re
import warnings

MIN_BRAINPY = (2, 7, 6)   # keep in sync with pyproject brainpy>=… (CI-enforced)

def _parse_release(version_str):
    m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", version_str)
    if m is None:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))

def check_brainpy_compatibility(min_version=MIN_BRAINPY):
    from importlib.metadata import version, PackageNotFoundError
    try:
        brainpy_version = version("brainpy")
    except PackageNotFoundError:
        return  # brainpy not installed — tolerated (contract B)
    release = _parse_release(brainpy_version)
    if release is None:
        warnings.warn(
            f"brainpy.state could not parse the installed brainpy version "
            f"{brainpy_version!r}; skipping the compatibility check.",
            stacklevel=2,
        )
        return
    if release < tuple(min_version):
        req = ".".join(map(str, min_version))
        raise ImportError(
            f"brainpy.state requires brainpy >= {req}, but found brainpy "
            f"{brainpy_version}. Upgrade with: pip install -U \"brainpy>={req}\""
        )
```

### `../../brainpy_state/__init__.py` (edit)
Replace the inline function + call + del (lines 17–41) with:
```python
from ._compat import check_brainpy_compatibility
check_brainpy_compatibility()
del check_brainpy_compatibility
```

### `../../brainpy_state/_compat_test.py` (new)
- Pure `_parse_release`: `2.7.6`, `2.7`, `2.7.6.post1`, `2.7.6rc1`, `2.7.6.dev0`,
  `2.7.6+cuda12`, `garbage`/`""`→None.
- `check_brainpy_compatibility` (mock `importlib.metadata.version`):
  absent→no raise/no warn; `2.7.6`/`2.7.8`→ok; `2.7.5`/`2.6.9`/`2.7`→ImportError;
  `2.7.6.dev0`→ok; `garbage`→warns; message contains installed version + `2.7.6`;
  injectable `min_version=(99,0,0)`→ImportError.
- Consistency: `tomllib` load pyproject, regex `brainpy>=X.Y.Z`, assert ==MIN_BRAINPY.

## Edge cases
- `"2.7"` → `(2,7,0)` < threshold → blocked (intended).
- `.dev0`/`rc1`/`.post1`/`+local` of target → allowed (release-segment compare).
- brainpy absent → silent return (no raise, no warn).
- unparseable-but-present → warn + allow (can't prove incompatibility).
- internal bug → propagates (no catch-all).
- `del` of the imported name in `__init__` is harmless (real def lives in `_compat`).
