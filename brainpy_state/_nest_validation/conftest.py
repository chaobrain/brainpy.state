# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""pytest configuration for the live-NEST validation harness.

Registers the ``requires_nest`` marker so ``pytest -m requires_nest`` selects the
live-NEST parity tests and no ``PytestUnknownMarkWarning`` is emitted. The actual
skip-when-absent behaviour lives in
:func:`brainpy_state._nest_validation.nest_compare.requires_nest`.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_nest: test needs a live NEST install (skipped when `import nest` fails)",
    )
