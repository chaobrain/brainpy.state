# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Attach the gate markers (``k0_gate``, ``k0_g1`` ..., ``k1a_gate``, ``k1a_g1`` ...) to the gate tests.

The registry is :data:`brainspike._kernel.validation.gates.GATES`; see its module
docstring. ``pytest -m k0_gate brainspike/_kernel`` runs the whole gate suite.
"""
from __future__ import annotations

import pathlib

import brainstate
import jax
import pytest

from brainspike._kernel.validation.gates import gates_for

_ROOT = pathlib.Path(__file__).resolve().parent


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    for item in items:
        try:
            rel = pathlib.Path(item.path).resolve().relative_to(_ROOT).as_posix()
        except ValueError:  # pragma: no cover - items outside brainspike/_kernel
            continue
        parts = item.nodeid.split('::', 1)
        node = rel + ('::' + parts[1] if len(parts) == 2 else '')
        gates = gates_for(node)
        if gates:
            for suite in sorted({g.suite for g in gates}):
                item.add_marker(getattr(pytest.mark, suite))
            for g in gates:
                item.add_marker(getattr(pytest.mark, g.marker))


# Kernel tests pin JAX's default ``jax_enable_x64=False`` behaviour (float32 dtypes,
# no-retrace checks, module-level float32 constants). Many legacy test modules (e.g.
# ``_nest_neuron/*_test.py``) call ``jax.config.update('jax_enable_x64', True)`` at
# import time; under ``pytest-xdist`` every worker imports every test module during
# collection, so that switch leaks into the whole process and the kernel tests would
# run in x64 mode depending on what else was collected. The two hooks below restore
# the default for kernel test modules, both while they are imported (module-level
# constants) and while they run. Tests that need x64 still opt in locally with
# ``jax.enable_x64(True)``, which nests inside this.


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    if isinstance(collector, pytest.Module):
        with jax.enable_x64(False):
            yield
    else:
        yield


@pytest.fixture(autouse=True, scope='module')
def _kernel_default_x64():
    """Run every kernel test module under JAX's default ``jax_enable_x64=False``.

    Also restores the global ``brainstate`` precision afterwards, so a kernel test
    that changes it cannot leak into the legacy tests (which rely on the
    ``precision=64`` they set at import) running later on the same worker.
    """
    precision = brainstate.environ.get_precision()
    try:
        with jax.enable_x64(False):
            yield
    finally:
        brainstate.environ.set(precision=precision)
