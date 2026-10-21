# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of ``timed_event`` network delivery (K0 T7): rings, per-edge delivery, kind
conversions, capacity checks, overflow counting, batching and gradients."""
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainpy_state._kernel.connectivity import all_to_all, one_to_one
from brainpy_state._kernel.models import (
    iaf_psc_alpha,
    iaf_psc_exp_ps,
    poisson_generator_ps,
    spike_generator,
    spike_generator_ps,
)
from brainpy_state._kernel.network import Network
from brainpy_state._kernel.ports import PortError, TimedEvent, TimedEventPayload
from brainpy_state._kernel.timed import TimedDelivery, TimedRing, accumulate, grid_to_timed

DT = 0.1


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def simulate(net, T, *recs, precision='float64', batch=None):
    res = net.compile(precision).runner(batch=batch).simulate(T * u.ms)
    return [res[r] for r in recs]


def times_of(rec, col=0):
    sp = np.asarray(rec['spike'])[:, col]
    idx = np.nonzero(sp)[0]
    return idx * DT + np.asarray(rec['spike_offset'].to_decimal(u.ms))[idx, col]


def V_of(rec):
    return np.asarray(rec['V_m'].to_decimal(u.mV))


def direct_run(arrivals, n_steps, params=None, capacity=16):
    """Single iaf_psc_exp_ps driven directly with events at absolute arrival times."""
    from brainpy_state._kernel.models.precise_test import run_neuron
    neu = iaf_psc_exp_ps.with_capacity(capacity)(1, params=params, precision='float64')
    return run_neuron(neu, n_steps, [(t, w, 0) for t, w in arrivals])


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------

class TestTimedRing:
    def test_write_read_per_edge_delays(self, x64):
        r = TimedRing(n=3, length=4, capacity=2, dtype=np.float64)
        buf = r.init()
        assert buf['offsets'].shape == (4, 3, 2) and np.isinf(buf['offsets']).all()
        for step in range(6):
            val = {'offsets': jnp.full((3, 2), 0.01 * (step + 1)), 'weights': jnp.full((3, 2), step)}
            buf = r.write(buf, step, val)
        o, w = r.read_edges(buf, 6, np.array([1, 3, 4]), np.array([0, 2, 1]))
        np.testing.assert_allclose(o[:, 0], [0.06, 0.04, 0.03])
        np.testing.assert_allclose(w[:, 1], [5, 3, 2])
        assert r.nbytes() == 2 * 4 * 3 * 2 * 8
        assert 'capacity=2' in repr(r)
        with pytest.raises(ValueError, match='outside'):
            r.read_edges(buf, 6, np.array([5]), np.array([0]))
        with pytest.raises(ValueError, match='length'):
            TimedRing(1, 0, 1, np.float32)

    def test_in_edge_table(self):
        post = np.array([2, 0, 2, 2, 1])

        class Seg:  # minimal segment stand-in
            capacity = 1
        d = TimedDelivery([], post, n_post=4, target='timed', capacity=4)
        assert d.indegree.tolist() == [1, 1, 3, 0]
        assert d.table.shape == (4, 3)
        assert sorted(d.table[2].tolist()) == [0, 2, 3] and d.table[3].tolist() == [5, 5, 5]

    def test_grid_to_timed_and_accumulate(self, x64):
        p = grid_to_timed(jnp.array([0.0, 2.5, -1.0]), DT, 3, jnp.float64)
        assert p.valid.tolist() == [[False] * 3, [True, False, False], [True, False, False]]
        np.testing.assert_allclose(p.offsets.to_decimal(u.ms)[1:, 0], [DT, DT])

        class Seg:
            pos = slice(0, 2)
            local = np.array([2, 0])
            full = False
        acc = accumulate(None, p, Seg, 3, 10.0, 2, DT, jnp.float64)  # rows 0,1 -> locals 2,0
        assert acc.valid[0, 0] and not acc.valid[2, 0]
        np.testing.assert_allclose(u.get_mantissa(acc.weights)[0, 0], 25.0)


# ---------------------------------------------------------------------------
# network delivery
# ---------------------------------------------------------------------------

class TestDelivery:
    TIMES = [0.537, 2.004, 2.033, 5.1, 7.77]

    @pytest.mark.parametrize('delay', [0.0, 0.1, 0.25, 1.5])
    def test_delay_on_grid_offsets_preserved(self, x64, delay):
        net = Network(dt=DT * u.ms)
        g = net.add(spike_generator_ps, 1, params=dict(spike_times=self.TIMES * u.ms))
        n = net.add(iaf_psc_exp_ps, 1)
        net.connect(g, n, port='syn', weight=150.0 * u.pA, delay=delay * u.ms)
        rec, = simulate(net, 20.0, net.observe(n, 'V_m'))
        d_steps = int(np.floor(delay / DT + 0.5))  # NEST rounding of delays
        ref = direct_run([(t + d_steps * DT, 150.0) for t in self.TIMES], 200)
        np.testing.assert_allclose(V_of(rec)[:, 0], ref['V_m'][:, 0], atol=1e-12)

    def test_heterogeneous_per_edge_delays_and_partial_views(self, x64):
        net = Network(dt=DT * u.ms)
        g = net.add(spike_generator_ps, 2, params=dict(spike_times=[[0.31] * u.ms, [0.77] * u.ms]))
        n = net.add(iaf_psc_exp_ps, 3)
        # edges (pre, post): (0, 2) d=0.3, (1, 2) d=1.0, (1, 0) d=0.5 ; post view reversed
        net.connect(g, n[[2, 0]], rule=all_to_all(), port='syn',
                    weight=[100.0, 0.0, 50.0, 70.0] * u.pA, delay=[0.3, 0.2, 1.0, 0.5] * u.ms)
        rec, = simulate(net, 5.0, net.observe(n, 'V_m'))
        V = V_of(rec)
        # EdgeSet order is post-major: (pre0,post0=n2) (pre1,post0=n2) (pre0,post1=n0) (pre1,post1=n0)
        ref2 = direct_run([(0.31 + 0.3, 100.0), (0.77 + 0.2, 0.0)], 50)
        ref0 = direct_run([(0.31 + 1.0, 50.0), (0.77 + 0.5, 70.0)], 50)
        np.testing.assert_allclose(V[:, 2], ref2['V_m'][:, 0], atol=1e-12)
        np.testing.assert_allclose(V[:, 0], ref0['V_m'][:, 0], atol=1e-12)
        assert np.all(V[:, 1] == -70.0)

    def test_concatenated_sources_and_merge_of_projections(self, x64):
        net = Network(dt=DT * u.ms)
        a = net.add(spike_generator_ps, 1, params=dict(spike_times=[1.05, 3.0] * u.ms), name='a')
        b = net.add(spike_generator_ps.with_capacity(2), 1, params=dict(spike_times=[1.02] * u.ms), name='b')
        c = net.add(spike_generator_ps.with_capacity(4), 1, params=dict(spike_times=[1.08] * u.ms),
                    name='c')
        n = net.add(iaf_psc_exp_ps, 1)
        net.connect(a + b, n, port='syn', weight=80.0 * u.pA, delay=0.1 * u.ms)
        net.connect(c, n, port='syn', weight=-30.0 * u.pA, delay=0.1 * u.ms)
        rec, = simulate(net, 6.0, net.observe(n, 'V_m', 'overflow'))
        ref = direct_run([(1.15, 80.0), (3.1, 80.0), (1.12, 80.0), (1.18, -30.0)], 60)
        np.testing.assert_allclose(V_of(rec)[:, 0], ref['V_m'][:, 0], atol=1e-12)
        assert int(np.asarray(rec['overflow']).max()) == 0

    def test_grid_event_into_timed_port_has_offset_dt(self, x64):
        net = Network(dt=DT * u.ms)
        g = net.add(spike_generator, 1, params=dict(spike_times=[1.0, 1.0, 2.5] * u.ms))
        n = net.add(iaf_psc_exp_ps, 1)
        net.connect(g, n, port='syn', weight=100.0 * u.pA, delay=0.5 * u.ms)
        rec, = simulate(net, 5.0, net.observe(n, 'V_m'))
        ref = direct_run([(1.5, 200.0), (3.0, 100.0)], 50)  # NEST stamp of a grid spike
        np.testing.assert_allclose(V_of(rec)[:, 0], ref['V_m'][:, 0], atol=1e-12)

    def test_timed_source_into_grid_port_drops_offsets(self, x64):
        def run(model, times):
            net = Network(dt=DT * u.ms)
            g = net.add(model, 1, params=dict(spike_times=times * u.ms))
            n = net.add(iaf_psc_alpha, 1)
            net.connect(g, n, port='ex', weight=300.0 * u.pA, delay=1.0 * u.ms)
            rec, = simulate(net, 8.0, net.observe(n, 'V_m'))
            return V_of(rec)
        precise = run(spike_generator_ps, [0.52, 0.58, 3.31])
        grid = run(spike_generator, [0.6, 0.6, 3.4])
        np.testing.assert_allclose(precise, grid, atol=1e-12)

    def test_neuron_to_neuron_precise_chain(self, x64):
        net = Network(dt=DT * u.ms)
        a = net.add(iaf_psc_exp_ps, 1, params=dict(I_e=600.0 * u.pA), name='a')
        b = net.add(iaf_psc_exp_ps, 1, name='b')
        net.connect(a, b, port='syn', weight=500.0 * u.pA, delay=0.7 * u.ms)
        ra, rb = simulate(net, 40.0, net.observe(a, 'spike', 'spike_offset'),
                          net.observe(b, 'V_m', 'I_syn_ex'))
        ta = times_of(ra)
        assert len(ta) >= 2
        ref = direct_run([(t + 0.7, 500.0) for t in ta], 400)
        np.testing.assert_allclose(V_of(rb)[:, 0], ref['V_m'][:, 0], atol=1e-12)
        # spike events of a timed output are event recordables
        t, gids, counts = ra.spike_events()
        np.testing.assert_allclose(np.asarray(t.to_decimal(u.ms)), np.ceil(np.round(ta / DT, 6)) * DT)

    def test_runner_step_returns_timed_payload(self, x64):
        net = Network(dt=DT * u.ms)
        g = net.add(spike_generator_ps, 1, params=dict(spike_times=[0.05] * u.ms), name='g')
        n = net.add(iaf_psc_exp_ps, 1, name='n')
        net.connect(g, n, port='syn', weight=10.0 * u.pA, delay=0.0 * u.ms)
        run = net.compile('float64').runner()
        out = run.step()
        p = out['g']['spike']
        assert isinstance(p, TimedEventPayload)
        assert float(p.offsets.to_decimal(u.ms)[0, 0]) == pytest.approx(0.05)
        assert isinstance(out['n']['spike'], TimedEventPayload)


class TestCapacity:
    def test_connect_time_source_capacity(self):
        net = Network(dt=DT * u.ms)
        g = net.add(spike_generator_ps, 1, params=dict(spike_times=[1.0] * u.ms))
        n = net.add(iaf_psc_exp_ps.with_capacity(4), 1)
        with pytest.raises(PortError, match='capacity is 4'):
            net.connect(g, n, port='syn', weight=1.0 * u.pA)
        net.connect(g, n, port='syn', weight=1.0 * u.pA, strict_capacity=False)

    def test_compile_time_worst_case_and_counted_overflow(self, x64):
        def build(strict):
            net = Network(dt=DT * u.ms)
            g = net.add(spike_generator_ps.with_capacity(1), 3,
                        params=dict(spike_times=[[1.01] * u.ms, [1.03] * u.ms, [1.02] * u.ms]))
            n = net.add(iaf_psc_exp_ps.with_capacity(2), 1)
            net.connect(g, n, port='syn', weight=50.0 * u.pA, delay=0.1 * u.ms, strict_capacity=strict)
            return net, n
        net, _ = build(True)
        with pytest.raises(PortError, match='strict_capacity=False'):
            net.compile('float64')
        net, n = build(False)
        rec, = simulate(net, 3.0, net.observe(n, 'V_m', 'overflow'))
        ovf = np.asarray(rec['overflow'])[:, 0]
        assert ovf[-1] == 1 and int(np.argmax(ovf)) == 11  # spikes in step 10, delay 1 step
        ref = direct_run([(1.11, 50.0), (1.12, 50.0)], 30)   # the latest event is dropped
        np.testing.assert_allclose(V_of(rec)[:, 0], ref['V_m'][:, 0], atol=1e-12)

    def test_grid_sources_count_one_event_per_step(self):
        net = Network(dt=DT * u.ms)
        g = net.add(spike_generator, 5, params=dict(spike_times=[1.0] * u.ms))
        n = net.add(iaf_psc_exp_ps.with_capacity(1), 2)
        net.connect(g, n, port='syn', weight=1.0 * u.pA)   # 5 grid sources -> 1 summed event
        net.compile()

    def test_dense_storage_rejected_and_describe(self):
        net = Network(dt=DT * u.ms)
        g = net.add(poisson_generator_ps, 4, params=dict(rate=10.0 * u.Hz))
        n = net.add(iaf_psc_exp_ps, 4)
        with pytest.raises(ValueError, match='CSR'):
            net.connect(g, n, one_to_one(), port='syn', weight=1.0 * u.pA, storage='dense')
        net.connect(g, n, one_to_one(), port='syn', weight=1.0 * u.pA, delay=2.0 * u.ms)
        text = net.compile().describe(print_=False)
        assert 'payload=timed' in text and 'buffer=timed-side' in text
        assert '20 x 4 x 8 (offsets, weights)' in text
        assert 'precise_spike_times=True' in text


class TestBatchingAndGradients:
    def test_batched_poisson_trials(self):
        net = Network(dt=DT * u.ms, seed=3)
        g = net.add(poisson_generator_ps, 2, params=dict(rate=800.0 * u.Hz))
        n = net.add(iaf_psc_exp_ps, 2)
        net.connect(g, n, one_to_one(), port='syn', weight=200.0 * u.pA, delay=0.5 * u.ms)
        rec_g = net.observe(g, 'spike', 'spike_offsets')
        rec_n = net.observe(n, 'V_m')
        res = net.compile().runner(batch=3).simulate(20.0 * u.ms)
        V = np.asarray(res[rec_n]['V_m'].to_decimal(u.mV))
        assert V.shape == (3, 200, 2)
        assert not np.allclose(V[0], V[1])
        res1 = net.compile().runner().simulate(20.0 * u.ms)   # unbatched == trial 0
        np.testing.assert_allclose(np.asarray(res1[rec_n]['V_m'].to_decimal(u.mV)), V[0], atol=1e-5)
        offs = np.asarray(res[rec_g]['spike_offsets'].to_decimal(u.ms))
        assert offs.shape == (3, 200, 2, 8)

    def test_gradient_through_timed_delivery(self, x64):
        from brainpy_state._kernel.runner import Program
        net = Network(dt=DT * u.ms)
        g = net.add(spike_generator_ps, 1, params=dict(spike_times=[0.43, 1.27] * u.ms))
        n = net.add(iaf_psc_exp_ps, 1)
        p = net.connect(g, n, port='syn', weight=90.0 * u.pA, delay=0.3 * u.ms, trainable=True)
        rec = net.observe(n, 'V_m')
        plan = net.compile('float64')
        prog = Program(plan)
        params, state = prog.read_params(), prog.init_state()

        def loss(w):
            ps = {**params, 'weights': {**params['weights'], p.name: [w]}}
            _, recs = prog.rollout(ps, state, 0, jnp.uint32(0), 30)
            return jnp.sum(recs[rec.name]['V_m'])

        w0 = params['weights'][p.name][0]
        grad = jax.grad(loss)(w0)
        eps = 1e-4
        unit = u.get_unit(w0)
        fd = u.get_mantissa(loss(w0 + eps * unit) - loss(w0 - eps * unit)) / (2 * eps)
        grad = u.get_mantissa(grad)
        np.testing.assert_allclose(grad, fd, rtol=1e-6)
        assert float(jnp.abs(grad).sum()) > 0
