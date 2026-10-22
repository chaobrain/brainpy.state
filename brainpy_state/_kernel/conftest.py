# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Attach the K0 gate markers (``k0_gate``, ``k0_g1`` ...) to the gate tests.

The registry is :data:`brainpy_state._kernel.validation.gates.GATES`; see its module
docstring. ``pytest -m k0_gate brainpy_state/_kernel`` runs the whole gate suite.
"""
from __future__ import annotations

import pathlib

import pytest

from brainpy_state._kernel.validation.gates import gates_for

_ROOT = pathlib.Path(__file__).resolve().parent


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    for item in items:
        try:
            rel = pathlib.Path(item.path).resolve().relative_to(_ROOT).as_posix()
        except ValueError:  # pragma: no cover - items outside brainpy_state/_kernel
            continue
        parts = item.nodeid.split('::', 1)
        node = rel + ('::' + parts[1] if len(parts) == 2 else '')
        gates = gates_for(node)
        if gates:
            item.add_marker(pytest.mark.k0_gate)
            for g in gates:
                item.add_marker(getattr(pytest.mark, g.marker))
