# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""The gate registry points at real tests, and the markers reach them."""
import importlib
import pathlib

import pytest

from brainpy_state._kernel.validation import gates as G
from brainpy_state._kernel.validation.markers import HAS_BRIAN2, requires_brian2

KERNEL = pathlib.Path(G.__file__).resolve().parent.parent


def _resolve(prefix):
    path, _, rest = prefix.partition('::')
    f = KERNEL / path
    assert f.is_file(), prefix
    if not rest:
        return
    mod = importlib.import_module('brainpy_state._kernel.' + path[:-3].replace('/', '.'))
    names = rest.split('::')
    obj = mod
    for name in names[:-1]:
        obj = getattr(obj, name)
    last = names[-1]
    found = [a for a in dir(obj) if a == last or (a.startswith(last) and a.startswith('test'))]
    assert found, prefix


@pytest.mark.parametrize('gate', G.GATES, ids=lambda g: g.id)
def test_every_prefix_resolves(gate):
    assert gate.id.startswith('G') and gate.criterion and gate.marker == 'k0_' + gate.id.lower()
    assert gate.tests or gate.command
    for p in gate.tests:
        _resolve(p)


def test_registry_covers_spec_gates():
    assert [g.id for g in G.GATES] == [f'G{k}' for k in range(1, 11)]
    assert G.marker_names()[:2] == ['k0_gate', 'k0_g1'] and len(G.marker_names()) == 11
    assert [g.id for g in G.gates_for('validation/stdp_nest_test.py::TestStdpVsNest::test_x')] == ['G8']
    both = G.gates_for('models/nest_parity_test.py::TestIafPscAlphaVsNest::test_psp_traces')
    assert [g.id for g in both] == ['G1']


def test_markers_attached(request):
    # this module is not a gate test; a known gate test file carries the markers
    assert request.node.get_closest_marker('k0_gate') is None


def test_requires_brian2_marker():
    @requires_brian2
    def f():
        pass  # pragma: no cover
    names = [m.name for m in f.pytestmark]
    assert names == ['skipif', 'requires_brian2']
    assert f.pytestmark[0].args[0] is (not HAS_BRIAN2)
