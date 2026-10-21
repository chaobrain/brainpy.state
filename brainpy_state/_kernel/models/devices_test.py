# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of the K0 devices: poisson_generator, spike_generator, dc_generator."""
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainpy_state._kernel.context import StepContext
from brainpy_state._kernel.models import dc_generator, poisson_generator, spike_generator
from brainpy_state._kernel.models._testing import rollout

DT = 0.1 * u.ms


def _ctx(i, key=None):
    return StepContext.create(i * DT, DT, i, key)


class TestPoissonGenerator:
    def test_rate_statistics(self):
        n, steps, rate = 2000, 500, 40.0  # 4 expected spikes per unit
        pg = poisson_generator(n, params=dict(rate=rate * u.Hz))
        out, _ = rollout(pg, steps, DT, key=jax.random.key(3))
        counts = np.asarray(out['spike'])
        assert out['spike'].dtype == jnp.float32
        per_unit = counts[1:].sum(0)  # step 0 is outside the window (t_min = 0 < n)
        lam = rate * 1e-3 * 0.1 * (steps - 1)
        se = np.sqrt(lam / n)
        assert abs(per_unit.mean() - lam) < 4 * se
        # Poisson: variance == mean (Fano factor 1)
        assert abs(per_unit.var() / per_unit.mean() - 1.0) < 0.1
        assert counts[0].sum() == 0

    def test_high_rate_gives_counts_above_one(self):
        pg = poisson_generator(500, params=dict(rate=20000.0 * u.Hz))  # lambda = 2 per step
        c = np.asarray(pg.update(_ctx(1, jax.random.key(0)))['spike'])
        assert c.max() > 1 and abs(c.mean() - 2.0) < 0.3
        assert pg.outputs['spike'].max_multiplicity is None

    def test_window_and_heterogeneous_rates(self):
        pg = poisson_generator(3, params=dict(rate=[0.0, 1e4, 1e4] * u.Hz, start=[0.0, 0.0, 1.0] * u.ms,
                                              stop=[10.0, 2.0, 10.0] * u.ms))
        out, _ = rollout(pg, 40, DT, key=jax.random.key(1))
        c = np.asarray(out['spike'])
        assert c[:, 0].sum() == 0
        assert c[21:, 1].sum() == 0 and c[1:21, 1].sum() > 0  # active iff 0 < n <= 20
        assert c[:11, 2].sum() == 0 and c[11:, 2].sum() > 0  # active iff 10 < n

    def test_origin_shifts_window(self):
        pg = poisson_generator(1, params=dict(rate=1e5 * u.Hz, stop=0.5 * u.ms, origin=1.0 * u.ms))
        out, _ = rollout(pg, 30, DT, key=jax.random.key(2))
        active = np.nonzero(np.asarray(out['spike'])[:, 0])[0]
        assert active.min() >= 11 and active.max() <= 15

    def test_requires_key(self):
        with pytest.raises(ValueError, match='ctx.key'):
            poisson_generator(1).update(_ctx(1))

    def test_deterministic_given_keys_and_vmap(self):
        pg = poisson_generator(100, params=dict(rate=5e3 * u.Hz))
        s, p = pg.read()

        @jax.jit
        def draw(key):
            return pg.step(s, p, {}, _ctx(1, key))[1]['spike']

        a = draw(jax.random.key(7))
        np.testing.assert_array_equal(a, draw(jax.random.key(7)))
        batch = jax.vmap(draw)(jax.random.split(jax.random.key(7), 4))
        assert batch.shape == (4, 100)
        assert not np.array_equal(np.asarray(batch[0]), np.asarray(batch[1]))

    def test_float64(self):
        with jax.enable_x64(True):
            pg = poisson_generator(10, params=dict(rate=1e4 * u.Hz), precision='float64')
            assert pg.update(_ctx(1, jax.random.key(0)))['spike'].dtype == jnp.float64


class TestSpikeGenerator:
    def test_duplicates_counted(self):
        """goal-01: three spikes at the same time are a count of 3 (NEST: three events)."""
        sg = spike_generator(1, params=dict(spike_times=[1.0, 1.0, 1.0, 2.5] * u.ms))
        out, _ = rollout(sg, 40, DT)
        c = np.asarray(out['spike'])[:, 0]
        assert c[9] == 3.0 and c[24] == 1.0 and c.sum() == 4.0  # stamp T -> step T/dt - 1

    def test_multiplicities_and_ragged_per_unit(self):
        sg = spike_generator(3, params=dict(
            spike_times=[[0.5, 1.0] * u.ms, [1.0] * u.ms, []], spike_multiplicities=[[2, 1], [4], []]))
        out, _ = rollout(sg, 20, DT)
        c = np.asarray(out['spike'])
        assert c[4, 0] == 2.0 and c[9, 0] == 1.0 and c[9, 1] == 4.0 and c[:, 2].sum() == 0

    def test_ragged_quantities_and_2d_array(self):
        sg = spike_generator(2, params=dict(spike_times=[[0.3] * u.ms, [0.3, 0.6] * u.ms]))
        c = np.asarray(rollout(sg, 10, DT)[0]['spike'])
        assert c[2].tolist() == [1.0, 1.0] and c[5].tolist() == [0.0, 1.0]
        sg2 = spike_generator(2, params=dict(spike_times=jnp.array([[0.2, 0.4], [0.4, 0.4]]) * u.ms,
                                             spike_multiplicities=jnp.array([[1, 1], [1, 2]])))
        c2 = np.asarray(rollout(sg2, 10, DT)[0]['spike'])
        assert c2[3].tolist() == [1.0, 3.0]

    def test_shared_train_with_shared_multiplicities(self):
        sg = spike_generator(2, params=dict(spike_times=[0.2, 0.4] * u.ms, spike_multiplicities=[3, 1]))
        c = np.asarray(rollout(sg, 10, DT)[0]['spike'])
        assert c[1].tolist() == [3.0, 3.0] and c[3].tolist() == [1.0, 1.0]

    def test_off_grid_rounds_up_and_time_zero_never_fires(self):
        sg = spike_generator(1, params=dict(spike_times=[0.0, 0.33, 0.5000001] * u.ms))
        c = np.asarray(rollout(sg, 10, DT)[0]['spike'])[:, 0]
        # 0.33 -> stamp 0.4 (step 3); 0.5000001 snaps to 0.5 (step 4); 0 never fires
        assert c.tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0, 0]

    def test_window(self):
        sg = spike_generator(1, params=dict(spike_times=[0.5, 1.0, 1.5, 2.0] * u.ms,
                                            start=0.5 * u.ms, stop=1.5 * u.ms))
        c = np.asarray(rollout(sg, 30, DT)[0]['spike'])[:, 0]
        # NEST: stamp S fires iff t_min < S <= t_max -> 1.0 and 1.5
        assert np.nonzero(c)[0].tolist() == [9, 14]

    def test_origin(self):
        sg = spike_generator(1, params=dict(spike_times=[0.5] * u.ms, origin=1.0 * u.ms))
        c = np.asarray(rollout(sg, 30, DT)[0]['spike'])[:, 0]
        assert np.nonzero(c)[0].tolist() == [14]

    def test_empty_and_errors(self):
        sg = spike_generator(2)
        assert np.asarray(rollout(sg, 5, DT)[0]['spike']).sum() == 0
        with pytest.raises(ValueError, match='without spike_times'):
            spike_generator(1, params=dict(spike_multiplicities=[1]))
        with pytest.raises(ValueError, match='trains for 3 units'):
            spike_generator(3, params=dict(spike_times=[[1.0] * u.ms, [2.0] * u.ms]))
        with pytest.raises(ValueError, match='shape'):
            spike_generator(1, params=dict(spike_times=[1.0, 2.0] * u.ms, spike_multiplicities=[1]))
        with pytest.raises(ValueError, match='non-negative integers'):
            spike_generator(1, params=dict(spike_times=[1.0] * u.ms, spike_multiplicities=[1.5]))
        with pytest.raises(ValueError, match='1-D'):
            spike_generator(1, params=dict(spike_times=jnp.zeros((1, 1, 1)) * u.ms))

    def test_for_loop_jit_float64(self):
        with jax.enable_x64(True):
            sg = spike_generator(1, params=dict(spike_times=[1000.0, 1000.1] * u.ms), precision='float64')
            out, _ = rollout(sg, 10005, DT)
            c = np.asarray(out['spike'])[:, 0]
            assert c.dtype == np.float64 and np.nonzero(c)[0].tolist() == [9999, 10000]

    def test_large_time_float32(self):
        sg = spike_generator(1, params=dict(spike_times=[1000.0, 1000.1] * u.ms))
        s, p = sg.read()
        step = jax.jit(lambda i: sg.step(s, p, {}, StepContext(t=i * DT, dt=DT, step=i))[1]['spike'][0])
        assert [float(step(k)) for k in (9998, 9999, 10000, 10001)] == [0.0, 1.0, 1.0, 0.0]


class TestDcGenerator:
    def test_nest_window_shift(self):
        dc = dc_generator(2, params=dict(amplitude=[100.0, -50.0] * u.pA, start=1.0 * u.ms, stop=2.0 * u.ms))
        out, _ = rollout(dc, 30, DT)
        I = np.asarray(out['I'].to_decimal(u.pA))
        # active iff t_min < n + 2 <= t_max: 10 < n + 2 <= 20 -> n in [9, 18]
        assert np.nonzero(I[:, 0])[0].tolist() == list(range(9, 19))
        np.testing.assert_array_equal(I[9:19, 1], -50.0)
        assert u.math.allclose(dc.dyn_states['I'].value, 0.0 * u.pA)

    def test_default_always_on_after_start(self):
        dc = dc_generator(1, params=dict(amplitude=10.0 * u.pA))
        I = np.asarray(rollout(dc, 5, DT)[0]['I'].to_decimal(u.pA))[:, 0]
        assert I.tolist() == [10.0] * 5

    def test_vmap_over_amplitude(self):
        dc = dc_generator(1)
        s, p = dc.read()
        f = jax.vmap(lambda a: dc.step(s, p.replace(amplitude=jnp.full(1, a) * u.pA), {}, _ctx(3))[1]['I'])
        out = f(jnp.array([1.0, 2.0]))
        np.testing.assert_array_equal(out.to_decimal(u.pA), [[1.0], [2.0]])
        assert dc.capabilities.differentiable is False and isinstance(dc, brainstate.nn.Module)
