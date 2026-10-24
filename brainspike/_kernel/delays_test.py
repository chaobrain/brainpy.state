# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of integer-step delays, delay classification and the two ring buffers."""
from __future__ import annotations

import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

from brainspike._kernel.delays import PostRing, PreRing, classify_delays, delay_steps
from brainspike._kernel.units import UnitMismatchError

DT = 0.1 * u.ms


class TestDelaySteps:
    def test_nest_rounding(self):
        # NEST: ld_round(ms * steps_per_ms) = floor(x * 10 + 0.5)
        d = [0.0, 0.04, 0.05, 0.1, 0.149, 0.15, 0.25, 1.5, 2.0] * u.ms
        assert delay_steps(d, DT).tolist() == [0, 0, 1, 1, 1, 2, 3, 15, 20]

    def test_float32_half_steps_round_up_like_nest(self):
        # 0.35 and 1.05 are not representable in float32 (0.34999999...); NEST gets the
        # float64 decimal and rounds the exact half up: 4 and 11 steps.
        d = u.Quantity(np.array([0.35, 1.05, 0.45, 0.3499], np.float32), unit=u.ms)
        assert delay_steps(d, DT).tolist() == [4, 11, 5, 3]
        dt32 = u.Quantity(np.float32(0.1), unit=u.ms)
        assert delay_steps(d, dt32).tolist() == [4, 11, 5, 3]
        assert delay_steps(np.array([0.35, 0.3499999]) * u.ms, DT).tolist() == [4, 3]

    def test_other_units_and_scalar(self):
        assert int(delay_steps(0.0015 * u.second, DT)) == 15
        assert delay_steps(1.0 * u.ms, 0.25 * u.ms).shape == ()

    @pytest.mark.parametrize('bad', [-0.1 * u.ms, [0.1, float('nan')] * u.ms, float('inf') * u.ms])
    def test_rejects_negative_and_nonfinite(self, bad):
        with pytest.raises(ValueError):
            delay_steps(bad, DT)

    def test_rejects_non_time(self):
        with pytest.raises(UnitMismatchError):
            delay_steps(1.0 * u.mV, DT)


class TestClassify:
    def test_scalar(self):
        lay = classify_delays(np.array([3, 3, 3]), np.array([0, 1, 1]), 2)
        assert lay.kind == 'scalar' and lay.steps == 3 and (lay.min, lay.max) == (3, 3)
        assert lay.values.tolist() == [3] and not lay.has_zero

    def test_per_source(self):
        lay = classify_delays(np.array([1, 4, 1, 4]), np.array([0, 2, 0, 2]), 3)
        assert lay.kind == 'per_source'
        # source 1 has no edge: it gets the minimum delay
        assert lay.steps.tolist() == [1, 1, 4] and lay.max == 4

    def test_per_edge(self):
        lay = classify_delays(np.array([0, 2, 5]), np.array([0, 0, 1]), 2)
        assert lay.kind == 'per_edge' and lay.has_zero and lay.values.tolist() == [0, 2, 5]

    def test_force(self):
        assert classify_delays(np.array([2, 2]), np.array([0, 1]), 2, force='per_edge').kind == 'per_edge'
        with pytest.raises(ValueError, match='not uniform'):
            classify_delays(np.array([1, 2]), np.array([0, 1]), 2, force='scalar')
        with pytest.raises(ValueError, match='same source'):
            classify_delays(np.array([1, 2]), np.array([0, 0]), 1, force='per_source')
        with pytest.raises(ValueError, match='unknown'):
            classify_delays(np.array([1]), np.array([0]), 1, force='ring')
        lay = classify_delays(np.array([1, 2]), np.array([0, 1]), 2, force='per_source')
        assert lay.kind == 'per_source'

    @pytest.mark.parametrize('force,kind', [(None, 'scalar'), ('per_edge', 'per_edge'),
                                            ('per_source', 'per_source')])
    def test_empty(self, force, kind):
        lay = classify_delays(np.zeros(0, int), np.zeros(0, int), 4, force=force)
        assert lay.kind == kind and lay.min == lay.max == 0


class TestPreRing:
    def test_write_read_wraparound(self):
        r = PreRing(2, 3, np.float32)
        buf = r.init()
        assert buf.shape == (3, 2) and r.nbytes() == 24
        for n in range(7):
            # before writing step n, delays 1..3 return steps n-1..n-3 (zeros before 0)
            for d in (1, 2, 3):
                exp = [n - d, 10 * (n - d)] if n - d >= 0 else [0, 0]
                np.testing.assert_array_equal(r.read(buf, n, d), exp)
            buf = r.write(buf, n, jnp.array([n, 10 * n], jnp.float32))

    def test_per_source(self):
        r = PreRing(3, 4, np.float32)
        buf = r.init()
        for n in range(5):
            buf = r.write(buf, n, jnp.array([n, 10 * n, 100 * n], jnp.float32))
        # at step 5: source 0 delay 1 -> step 4, source 2 delay 3 -> step 2, column subset
        out = r.read_per_source(buf, 5, np.array([1, 3]), np.array([0, 2]))
        np.testing.assert_array_equal(out, [4, 200])

    def test_errors(self):
        with pytest.raises(ValueError):
            PreRing(2, 0, np.float32)
        r = PreRing(2, 2, np.float32)
        with pytest.raises(ValueError):
            r.read(r.init(), 0, 3)
        with pytest.raises(ValueError):
            r.read(r.init(), 0, 0)
        with pytest.raises(ValueError):
            r.read_per_source(r.init(), 0, np.array([0]), np.array([0]))
        assert 'PreRing' in repr(r)


class TestPostRing:
    def test_deposit_read_clear(self):
        r = PostRing(2, 3, np.float32)
        buf = r.init()
        # at step 0 deposit for steps 1 and 3; at step 1 deposit for step 3 again
        buf = r.clear(buf, 0)
        buf = r.deposit(buf, 0, 1, jnp.array([1.0, 0.0]))
        buf = r.deposit(buf, 0, 3, jnp.array([0.0, 2.0]))
        np.testing.assert_array_equal(r.read(buf, 1), [1, 0])
        buf = r.clear(buf, 1)
        buf = r.deposit(buf, 1, 2, jnp.array([0.0, 3.0]))
        np.testing.assert_array_equal(r.read(buf, 2), [0, 0])
        np.testing.assert_array_equal(r.read(buf, 3), [0, 5])
        assert r.nbytes() == 24 and 'PostRing' in repr(r)

    def test_errors(self):
        with pytest.raises(ValueError):
            PostRing(1, 0, np.float32)
        r = PostRing(1, 2, np.float32)
        with pytest.raises(ValueError):
            r.deposit(r.init(), 0, 0, jnp.zeros(1))
        with pytest.raises(ValueError):
            r.deposit(r.init(), 0, 3, jnp.zeros(1))
