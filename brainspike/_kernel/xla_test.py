# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""The kernel's per-jit XLA compiler options (K1a-9) reach its jits and only its jits."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainspike._kernel import xla
from brainspike._kernel.connectivity import fixed_indegree
from brainspike._kernel.models import iaf_psc_alpha, poisson_generator
from brainspike._kernel.network import Network

BAD = {'xla_gpu_redzone_padding_bytes': 'not-an-int'}


def _small_runner():
    net = Network(dt=0.1 * u.ms, seed=0)
    pg = net.add(poisson_generator, 20, params=dict(rate=50.0 * u.Hz), name='pg')
    a = net.add(iaf_psc_alpha, 10, name='a')
    net.connect(pg, a, fixed_indegree(4), port='ex', weight=50.0 * u.pA, name='p')
    rec = net.observe(a, 'spike', name='rec')
    return net.compile().runner(), rec


def test_default_redzone_is_small():
    assert xla.KERNEL_COMPILER_OPTIONS == {'xla_gpu_redzone_padding_bytes': 64 * 1024}
    opts = xla.compiler_options()
    opts['x'] = 1  # a copy
    assert 'x' not in xla.KERNEL_COMPILER_OPTIONS


def test_kernel_jit_forwards_options_and_plain_jit_is_unaffected(monkeypatch):
    # an invalid value is rejected by XLA at compile time: proof that the option is passed
    monkeypatch.setattr(xla, 'KERNEL_COMPILER_OPTIONS', dict(BAD))
    with pytest.raises(Exception, match='xla_gpu_redzone_padding_bytes'):
        xla.kernel_jit(lambda x: x * 2.0)(jnp.ones(3))
    with pytest.raises(Exception, match='xla_gpu_redzone_padding_bytes'):
        xla.kernel_jax_jit()(lambda x: x * 3.0)(jnp.ones(3))
    # per jit: the user's jits in the same process compile with XLA's defaults
    np.testing.assert_array_equal(jax.jit(lambda x: x * 4.0)(jnp.ones(3)), 4.0 * np.ones(3))
    monkeypatch.setattr(xla, 'KERNEL_COMPILER_OPTIONS', {})
    assert float(xla.kernel_jit(lambda x: x + 1.0)(jnp.float32(1.0))) == 2.0


def test_runner_programs_carry_the_options(monkeypatch):
    monkeypatch.setattr(xla, 'KERNEL_COMPILER_OPTIONS', dict(BAD))
    run, _ = _small_runner()
    with pytest.raises(Exception, match='xla_gpu_redzone_padding_bytes'):
        run.cont(1.0 * u.ms)
    monkeypatch.undo()
    run, rec = _small_runner()
    res = run.cont(5.0 * u.ms)
    assert np.asarray(res[rec]['spike']).shape == (50, 10)


def test_options_do_not_change_results(monkeypatch):
    run, rec = _small_runner()
    a = np.asarray(run.cont(20.0 * u.ms)[rec]['spike'])
    monkeypatch.setattr(xla, 'KERNEL_COMPILER_OPTIONS', {})
    run2, rec2 = _small_runner()
    b = np.asarray(run2.cont(20.0 * u.ms)[rec2]['spike'])
    np.testing.assert_array_equal(a, b)


_GPU_SCRIPT = textwrap.dedent('''
    import os, sys
    os.environ['BRAINSPIKE_ALLOW_DIRECT_IMPORT'] = '1'
    sys.path.insert(0, {root!r})
    import jax, jax.numpy as jnp, brainunit as u
    from brainspike._kernel.validation.brunel import build_brunel
    dev = jax.devices()[0]
    MiB = 2 ** 20
    net, info = build_brunel(200)
    run = net.compile('float32').runner()
    res = run.cont(100.0 * u.ms)
    jax.block_until_ready(res[info['rec_E']]['spike'])
    kernel_peak = dev.memory_stats()['peak_bytes_in_use'] / MiB
    a = jnp.ones((512, 512)); b = jnp.ones((512, 512))
    jax.block_until_ready(jax.jit(lambda x, y: jnp.tanh(x @ y) + 1.0)(a, b))
    user_peak = dev.memory_stats()['peak_bytes_in_use'] / MiB
    print('PEAKS', kernel_peak, user_peak)
''')


@pytest.mark.skipif(not any(d.platform == 'gpu' for d in jax.devices()),
                    reason='needs a GPU (autotuning scratch exists only on GPU)')
def test_gpu_peak_small_for_kernel_and_default_for_user_jit():  # pragma: no cover - GPU only
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = subprocess.run([sys.executable, '-c', _GPU_SCRIPT.format(root=root)],
                         capture_output=True, text=True, timeout=600)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith('PEAKS')]
    assert line, out.stderr[-2000:]
    kernel_peak, user_peak = map(float, line[0].split()[1:])
    # Brunel order 200 is ~5-15 MiB without the 8 MiB redzones (~65 MiB with them)
    assert kernel_peak < 30.0
    # the user's autotuned matmul still gets XLA's default 8 MiB redzones
    assert user_peak - kernel_peak > 16.0
