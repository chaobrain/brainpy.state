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


@pytest.mark.parametrize('gate', G.ALL_GATES, ids=lambda g: g.id)
def test_every_prefix_resolves(gate):
    if gate.id.startswith('K1a-'):
        assert gate.marker == 'k1a_' + gate.id[4:].lower() and gate.suite == 'k1a_gate'
    else:
        assert gate.id.startswith('G') and gate.marker == 'k0_' + gate.id.lower()
        assert gate.suite == 'k0_gate'
    assert gate.criterion
    assert gate.tests or gate.command
    for p in gate.tests:
        _resolve(p)


def test_registry_covers_spec_gates():
    assert [g.id for g in G.GATES] == [f'G{k}' for k in range(1, 11)]
    names = G.marker_names()
    assert names[:2] == ['k0_gate', 'k0_g1'] and len(names) == 11 + 7
    assert names[11:13] == ['k1a_gate', 'k1a_g1'] and names[-1] == 'k1a_g6'
    assert [g.id for g in G.gates_for('validation/stdp_nest_test.py::TestStdpVsNest::test_x')] == ['G8']
    both = G.gates_for('models/nest_parity_test.py::TestIafPscAlphaVsNest::test_psp_traces')
    assert [g.id for g in both] == ['G1']


def test_k1a_registry_covers_spec_gates():
    assert [g.id for g in G.K1A_GATES] == [f'K1a-G{k}' for k in range(1, 7)]
    assert [g.id for g in G.gates_for('validation/k1a_g5_ai_nest_test.py::test_x')] == ['K1a-G5']
    assert [g.id for g in G.gates_for('validation/compile_gate_test.py::test_compiles')] == ['K1a-G1']
    assert G.gates_for('validation/g2_brunel_nest_test.py::t')[0].suite == 'k0_gate'


def test_pyproject_registers_every_marker():
    import tomllib
    root = KERNEL.parent.parent
    cfg = tomllib.loads((root / 'pyproject.toml').read_text())
    registered = {m.split(':', 1)[0] for m in cfg['tool']['pytest']['ini_options']['markers']}
    assert set(G.marker_names()) <= registered


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
