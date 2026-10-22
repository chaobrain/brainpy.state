# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""K1a-1 compile-time gate: every K0 network compiles in <= 5 s at default substeps.

Runs ``benchmarks/k1a_compile_time.py one <case>`` in a fresh process per network
(compile time = first ``simulate`` minus a warm ``cont``; first minus warm
``BPTT.grad`` for the LIF BPTT net). Wall-clock gates are host- and load-dependent, so
the test runs only when selected explicitly::

    python -m pytest -m k1a_compile brainpy_state/_kernel/validation/compile_gate_test.py

on an idle host (the record run is the remote host CPU, ``BENCH=1``; numbers in the
K1a-1 PR and the K1a report).
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[3] / 'benchmarks' / 'k1a_compile_time.py'
GATE_S = 5.0
CASES = [('brunel', []), ('coba', []), ('aeif', []), ('ps', ['--capacity', '64']),
         ('nmda', []), ('stdp', []), ('bptt_lif', [])]


@pytest.mark.k1a_compile
@pytest.mark.parametrize('case,args', CASES, ids=[c for c, _ in CASES])
def test_compiles_within_gate(request, case, args):
    if 'k1a_compile' not in (request.config.getoption('markexpr') or ''):
        pytest.skip('wall-clock gate: select with -m k1a_compile on an idle host')
    if not SCRIPT.exists():  # pragma: no cover - source checkout only
        pytest.skip(f'{SCRIPT} not found')
    env = dict(os.environ, JAX_PLATFORMS='cpu', BRAINPY_STATE_ALLOW_DIRECT_IMPORT='1')
    root = str(SCRIPT.parents[1])
    env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
    out = subprocess.run([sys.executable, str(SCRIPT), 'one', case, *args], capture_output=True,
                         text=True, timeout=900, env=env, cwd=root)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith('RESULT_JSON ')]
    assert lines, out.stderr[-2000:]
    r = json.loads(lines[-1][len('RESULT_JSON '):])
    print(f"{case}: compile {r['compile_s']:.2f} s (first {r['first_s']:.2f} s, warm {r['warm_s']:.4f} s)")
    assert r['compile_s'] <= GATE_S, r
