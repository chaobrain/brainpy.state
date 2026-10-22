# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tests of the runner: step contract, delays, schedule, batching, RNG, observers."""
from __future__ import annotations

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from brainpy_state._kernel.compiler import ScheduleError
from brainpy_state._kernel.connectivity import all_to_all, fixed_indegree, one_to_one
from brainpy_state._kernel.models import LIF, dc_generator, iaf_psc_alpha, poisson_generator, spike_generator
from brainpy_state._kernel.models.devices import poisson_counts
from brainpy_state._kernel.network import Network
from brainpy_state._kernel.rng import population_block_keys, step_block_keys
from brainpy_state._kernel.runner import Program, Runner, duration_steps, trial_indices
from brainpy_state._kernel.validation.brunel import build_brunel, rates

DT = 0.1 * u.ms
W = 2.0 * u.mV  # one event makes a LIF (V_th = 1 mV) spike in the step it arrives


def spike_steps(res, obs, col=None):
    """Steps (run-local) at which column ``col`` (or any column) spiked."""
    x = np.asarray(res[obs]['spike'])
    if col is not None:
        x = x[:, col]
    else:
        x = x.sum(-1)
    return np.nonzero(x)[0].tolist()


def lif(net, n, name):
    return net.add(LIF, n, name=name, spk_reset='hard')


def source(net, times_ms, name='sg'):
    return net.add(spike_generator, 1, params=dict(spike_times=list(times_ms) * u.ms), name=name)


# --------------------------------------------------------------------------- timing

class TestDeliveryTiming:
    @pytest.mark.parametrize('delay_ms,steps', [(0.1, 1), (1.5, 15), (0.15, 2), (0.0, 0), (0.04, 0)])
    def test_single_spike_arrival(self, delay_ms, steps):
        net = Network(dt=DT)
        sg = source(net, [1.0])  # stamp 10 -> emitted at step 9
        n = lif(net, 1, 'n')
        net.connect(sg, n, port='delta', weight=W, delay=delay_ms * u.ms)
        rec = net.observe(n, 'spike')
        res = net.compile().runner().simulate(5.0 * u.ms)
        assert spike_steps(res, rec) == [9 + steps]

    def test_zero_delay_chain_in_one_step(self):
        # inserted in reverse order: the schedule must sort them topologically
        net = Network(dt=DT)
        c = lif(net, 1, 'C')
        b = lif(net, 1, 'B')
        a = lif(net, 1, 'A')
        sg = source(net, [1.0])
        net.connect(b, c, port='delta', weight=W, delay=0.0 * u.ms)
        net.connect(a, b, port='delta', weight=W, delay=0.0 * u.ms)
        net.connect(sg, a, port='delta', weight=W, delay=0.0 * u.ms)
        rec = net.observe(a + b + c, 'spike')
        plan = net.compile()
        assert [plan.populations[i].name for i in plan.order] == ['sg', 'A', 'B', 'C']
        res = plan.runner().simulate(3.0 * u.ms)
        assert [spike_steps(res, rec, k) for k in range(3)] == [[9], [9], [9]]

    def test_unit_delay_chain(self):
        net = Network(dt=DT)
        sg = source(net, [1.0])
        a, b = lif(net, 1, 'A'), lif(net, 1, 'B')
        net.connect(sg, a, port='delta', weight=W)
        net.connect(a, b, port='delta', weight=W)
        rec = net.observe(a + b, 'spike')
        res = net.compile().runner().simulate(3.0 * u.ms)
        assert [spike_steps(res, rec, k) for k in range(2)] == [[10], [11]]

    @pytest.mark.parametrize('delays', [((0.0, 0.0),), ((0.0, 0.1), (0.1, 0.0)), ((0.0, 0.1), (0.0, 0.0))])
    def test_zero_delay_cycles_rejected(self, delays):
        net = Network(dt=DT)
        a, b = lif(net, 1, 'A'), lif(net, 1, 'B')
        pairs = [(a, a)] if len(delays) == 1 else [(a, b), (b, a)]
        for (pre, post), (d0, _) in zip(pairs, delays):
            net.connect(pre, post, port='delta', weight=W, delay=d0 * u.ms)
        if len(delays) == 2 and delays[1][0] == 0.0 and delays[0][0] == 0.0:
            with pytest.raises(ScheduleError, match='cycle'):
                net.compile()
        elif len(delays) == 1:
            with pytest.raises(ScheduleError, match='itself'):
                net.compile()
        else:
            net.compile()  # one zero-delay edge of the cycle has a delay: fine

    def test_zero_delay_cycle_message(self):
        net = Network(dt=DT)
        a, b, c = lif(net, 1, 'A'), lif(net, 1, 'B'), lif(net, 1, 'C')
        for pre, post in ((a, b), (b, c), (c, a)):
            net.connect(pre, post, port='delta', weight=W, delay=0.0 * u.ms)
        with pytest.raises(ScheduleError, match="D4"):
            net.compile()

    def test_cycle_with_unit_delay_runs(self):
        net = Network(dt=DT)
        sg = source(net, [1.0])
        a, b = lif(net, 1, 'A'), lif(net, 1, 'B')
        net.connect(sg, a, port='delta', weight=W)
        net.connect(a, b, port='delta', weight=W, delay=0.0 * u.ms)
        net.connect(b, a, port='delta', weight=W, delay=0.3 * u.ms)
        rec = net.observe(a + b, 'spike')
        res = net.compile().runner().simulate(2.0 * u.ms)
        # A at 10, B same step, back to A 3 steps later, ...
        assert spike_steps(res, rec, 0) == [10, 13, 16, 19]
        assert spike_steps(res, rec, 1) == [10, 13, 16, 19]

    def test_current_port_same_step_lif(self):
        # dc_generator emits from step 0; delay 1 step; LIF.I is latch=False (D-a): no extra step
        net = Network(dt=DT)
        dc = net.add(dc_generator, 1, params=dict(amplitude=5e8 * u.pA))  # 0.5 mA -> 0.5 mV
        n = lif(net, 1, 'n')
        net.connect(dc, n, port='I')
        rec = net.observe(n, 'V')
        res = net.compile('float32').runner().simulate(60.0 * u.ms)
        v = np.asarray(res[rec]['V'].to_decimal(u.mV))[:, 0]
        assert v[0] == 0.0 and v[1] > 0.0
        np.testing.assert_allclose(v[-1], 0.5, rtol=1e-4)  # pA -> mA conversion

    def test_current_port_one_step_latch_nest_model(self):
        # the same into a latched port (iaf_psc_alpha.I, NEST CurrentEvent): one more step
        net = Network(dt=DT)
        dc = net.add(dc_generator, 1, params=dict(amplitude=100.0 * u.pA))
        n = net.add(iaf_psc_alpha, 1, name='n', params=dict(V_th=1e3 * u.mV))
        net.connect(dc, n, port='I')
        rec = net.observe(n, 'V_m')
        res = net.compile('float32').runner().simulate(1.0 * u.ms)
        v = np.asarray(res[rec]['V_m'].to_decimal(u.mV))[:, 0]
        assert v[0] == v[1] == -70.0 and v[2] > -70.0


# --------------------------------------------------------------------------- delays

class TestDelayBuffers:
    def test_per_edge_delays_post_side(self):
        net = Network(dt=DT)
        sg = source(net, [1.0])
        n = lif(net, 4, 'n')
        p = net.connect(sg, n, all_to_all(), port='delta', weight=W,
                        delay=[0.1, 0.5, 0.0, 1.5] * u.ms)
        rec = net.observe(n, 'spike')
        plan = net.compile()
        pp = plan.projection(p)
        assert pp.placement == 'post' and pp.layout.kind == 'per_edge'
        assert [g.delay for g in pp.groups] == [0, 1, 5, 15] and pp.post_ring.length == 15
        res = plan.runner().simulate(4.0 * u.ms)
        assert [spike_steps(res, rec, k) for k in range(4)] == [[10], [14], [9], [24]]

    def test_per_source_delays_pre_side(self):
        net = Network(dt=DT)
        sg = net.add(spike_generator, 3, params=dict(spike_times=[0.5] * u.ms), name='sg')  # step 4
        n = lif(net, 2, 'n')
        d = np.tile([0.1, 0.3, 0.6], 2) * u.ms  # edges: post-major, pre ascending
        p = net.connect(sg, n, all_to_all(), port='delta', weight=W, delay=d)
        rec = net.observe(n, 'spike')
        plan = net.compile()
        pp = plan.projection(p)
        assert pp.placement == 'pre' and pp.layout.kind == 'per_source'
        assert plan.population('sg').rings['spike'].length == 6
        res = plan.runner().simulate(2.0 * u.ms)
        assert spike_steps(res, rec, 0) == [5, 7, 10] == spike_steps(res, rec, 1)

    def test_per_source_with_zero_delays(self):
        net = Network(dt=DT)
        sg = net.add(spike_generator, 2, params=dict(spike_times=[0.5] * u.ms))
        n = lif(net, 1, 'n')
        p = net.connect(sg, n, all_to_all(), port='delta', weight=W, delay=[0.0, 0.2] * u.ms)
        rec = net.observe(n, 'spike')
        plan = net.compile()
        assert plan.projection(p).layout.kind == 'per_source'
        assert spike_steps(plan.runner().simulate(2.0 * u.ms), rec) == [4, 6]

    @pytest.mark.parametrize('storage', ['dense', 'csr'])
    def test_post_side_equals_pre_side_for_equal_delays(self, storage):
        out = []
        for buf in ('pre', 'post'):
            net, info = build_brunel(20, seed=3, storage=storage, delay_buffer=buf)
            plan = net.compile()
            assert {p.placement for p in plan.projections} == {buf}
            res = plan.runner().simulate(100.0 * u.ms)
            out.append((np.asarray(res[info['rec_E']]['spike']), np.asarray(res[info['rec_I']]['spike'])))
        np.testing.assert_array_equal(out[0][0], out[1][0])
        np.testing.assert_array_equal(out[0][1], out[1][1])
        assert out[0][0].sum() > 0

    def test_distribution_delays(self):
        from brainpy_state._kernel.params import Uniform
        net = Network(dt=DT, seed=5)
        sg = source(net, [1.0])
        n = lif(net, 50, 'n')
        p = net.connect(sg, n, all_to_all(), port='delta', weight=W,
                        delay=Uniform(0.1 * u.ms, 2.0 * u.ms))
        rec = net.observe(n, 'spike')
        plan = net.compile()
        pp = plan.projection(p)
        assert pp.placement == 'post' and 1 <= pp.layout.min and pp.layout.max <= 20
        res = plan.runner().simulate(4.0 * u.ms)
        expected = 9 + np.floor(np.asarray(pp.edges.delay.to_decimal(u.ms)) * 10 + 0.5)
        got = [spike_steps(res, rec, k)[0] for k in range(50)]
        assert got == expected.astype(int).tolist()


# --------------------------------------------------------------------------- views

class TestViewsEndToEnd:
    def test_concatenated_views_both_sides(self):
        net = Network(dt=DT)
        sg = net.add(spike_generator, 2, params=dict(spike_times=[[0.4], [0.8]] * u.ms))
        a, b = lif(net, 2, 'A'), lif(net, 2, 'B')
        net.connect(sg[1] + sg[0], b[1] + a[0], one_to_one(), port='delta', weight=W)
        rec = net.observe(a + b, 'spike')
        res = net.compile().runner().simulate(2.0 * u.ms)
        assert [spike_steps(res, rec, k) for k in range(4)] == [[4], [], [], [8]]

    def test_overlapping_slices_autapses_on_global_ids(self):
        net = Network(dt=DT)
        a = lif(net, 20, 'A')
        p = net.connect(a[0:10], a[5:15], all_to_all(allow_autapses=False), port='delta',
                        weight=0.01 * u.mV)
        plan = net.compile()
        es = plan.edges(p)
        assert es.n_edges == 95 and not np.any(np.asarray(es.pre) == np.asarray(es.post))
        plan.runner().simulate(1.0 * u.ms)

    def test_zero_size_populations(self):
        net = Network(dt=DT)
        z = net.add(LIF, 0, name='Z')
        a = lif(net, 3, 'A')
        sg = source(net, [0.5])
        net.connect(z, a, port='delta', weight=W, source_port='spike')
        net.connect(a, z, port='delta', weight=W)
        net.connect(sg, a + z, port='delta', weight=W)
        rz = net.observe(z, 'spike', 'V')
        ra = net.observe(a + z, 'spike', interval=1.0 * u.ms)
        plan = net.compile()
        assert {p.placement for p in plan.projections[:2]} == {'none'}
        res = plan.runner().simulate(2.0 * u.ms)
        assert res[rz]['spike'].shape == (20, 0) and res[rz]['V'].shape == (20, 0)
        np.testing.assert_array_equal(np.asarray(res[ra]['spike']), [[1, 1, 1], [0, 0, 0]])


# --------------------------------------------------------------------------- runner

class TestRunner:
    def test_cont_does_not_retrace(self):
        net, info = build_brunel(10)
        plan = net.compile()
        run = plan.runner()
        run.simulate(20.0 * u.ms)
        for _ in range(3):
            run.cont(20.0 * u.ms)
        assert run.n_traces == 1 and run.n == 800
        # changing a parameter value does not retrace either
        tau = plan.population('E').model.param_states['tau_m']
        tau.value = tau.value * 2
        run.cont(20.0 * u.ms)
        assert run.n_traces == 1
        run.cont(10.0 * u.ms)  # a new duration compiles once more
        run.cont(10.0 * u.ms)
        assert run.n_traces == 2

    def test_simulate_equals_simulate_plus_cont(self):
        net, info = build_brunel(10, seed=7, record_v=True)
        plan = net.compile()
        whole = plan.runner().simulate(60.0 * u.ms)
        run = plan.runner()
        r1 = run.simulate(30.0 * u.ms)
        r2 = run.cont(30.0 * u.ms)
        for key in ('rec_E', 'rec_V'):
            name = 'spike' if key == 'rec_E' else 'V_m'
            a = np.asarray(u.get_mantissa(whole[info[key]][name]))
            b = np.concatenate([np.asarray(u.get_mantissa(r[info[key]][name])) for r in (r1, r2)])
            np.testing.assert_array_equal(a, b)
        assert float(r2.t_start.to_decimal(u.ms)) == 30.0 and r2.n_steps == 300
        np.testing.assert_allclose(np.asarray(r2[info['rec_E']].times.to_decimal(u.ms))[[0, -1]],
                                   [30.1, 60.0])

    def test_step_matches_cont(self):
        net, info = build_brunel(10, seed=2, record_v=True)
        plan = net.compile()
        a, b = plan.runner(), plan.runner()
        a.cont(1.0 * u.ms)
        for _ in range(10):
            out = b.step()
        assert set(out) == {'E', 'I', 'noise'} and out['E']['spike'].shape == (40,)
        for x, y in zip(jax.tree.leaves(a.state), jax.tree.leaves(b.state)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        assert b.n == 10 and float(b.t.to_decimal(u.ms)) == pytest.approx(1.0)
        b.reset()
        assert b.n == 0

    def test_runner_errors(self):
        net, _ = build_brunel(4)
        plan = net.compile()
        with pytest.raises(TypeError):
            plan.runner(learner=object())  # learners are no longer attached (D-c)
        with pytest.raises(TypeError, match='Program or a Plan'):
            Runner(object())
        with pytest.raises(ValueError):
            plan.runner(rng='shared')
        with pytest.raises(ValueError):
            plan.runner(batch=0)
        run = plan.runner()
        with pytest.raises(ValueError):
            run.cont(0.05 * u.ms)
        with pytest.raises(ValueError):
            run.cont(-1.0 * u.ms)
        assert duration_steps(0.0 * u.ms, DT) == 0

    @pytest.mark.parametrize('storage', ['csr', 'dense'])
    def test_structure_arrays_are_arguments_not_constants(self, storage):
        # closed-over index arrays would be embedded in the program as constants
        # (compile time and host memory proportional to the edge count)
        net, info = build_brunel(40, storage=storage)
        plan = net.compile()
        run = plan.runner()
        names = {f'{pp.name}/0' for pp in plan.projections}
        assert set(run._structure) == names
        assert set(run._structure['noise_EI/0']) == {'indices'}  # one_to_one: identity gather
        low = run._rollout.lower(run.program.read_params(), run.state, jnp.asarray(0, jnp.int32),
                                 run._trials, 10, run._structure)
        txt = low.as_text()
        n_edges = plan.projection('E_EI').n_edges
        shape = 'tensor<160x200xi1>' if storage == 'dense' else f'tensor<{n_edges}xi32>'
        consts = [ln for ln in txt.splitlines() if 'stablehlo.constant' in ln and shape in ln]
        assert consts == [] and shape in txt  # present as an argument only
        res = run.simulate(20.0 * u.ms)
        ref = plan.runner().simulate(20.0 * u.ms)
        np.testing.assert_array_equal(np.asarray(res[info['rec_E']]['spike']),
                                      np.asarray(ref[info['rec_E']]['spike']))

    def test_one_to_one_noise_equals_general_csr_path(self):
        # the gather fast path (identity one_to_one) is exact against the sparse kernel
        from brainpy_state._kernel.connectivity import storage as S
        net, info = build_brunel(10, seed=3, record_v=True)
        plan = net.compile('float64') if jax.config.jax_enable_x64 else net.compile()
        st = plan.projection('noise_EI').groups[0].storage
        assert st.identity
        a = plan.runner().simulate(30.0 * u.ms)
        st.gather = st.identity = False
        b = plan.runner().simulate(30.0 * u.ms)
        np.testing.assert_allclose(np.asarray(a[info['rec_V']]['V_m'].to_decimal(u.mV)),
                                   np.asarray(b[info['rec_V']]['V_m'].to_decimal(u.mV)),
                                   rtol=1e-6, atol=1e-5)
        assert isinstance(st, S.CSR)

    def test_brunel_rates_sane(self):
        net, info = build_brunel(200, seed=1)
        res = net.compile().runner().simulate(500.0 * u.ms)
        e, i = rates(res, info)
        # NEST and the old Simulator give about 57 Hz at order 200 (benchmarks/results)
        assert 40.0 < e < 75.0 and 40.0 < i < 75.0


class TestBatchingAndRng:
    def _spikes(self, res, info):
        return np.asarray(res[info['rec_E']]['spike'])

    def test_batch_trial0_equals_unbatched(self):
        net, info = build_brunel(10, seed=4)
        plan = net.compile()
        single = self._spikes(plan.runner().simulate(50.0 * u.ms), info)
        b1 = self._spikes(plan.runner(batch=1).simulate(50.0 * u.ms), info)
        b4 = self._spikes(plan.runner(batch=4).simulate(50.0 * u.ms), info)
        assert b4.shape == (4,) + single.shape
        np.testing.assert_array_equal(b1[0], single)
        np.testing.assert_array_equal(b4[0], single)
        assert not np.array_equal(b4[1], b4[0])
        c4 = self._spikes(plan.runner(batch=4, rng='common').simulate(50.0 * u.ms), info)
        for k in range(4):
            np.testing.assert_array_equal(c4[k], single)

    def test_batched_cont_and_step(self):
        net, info = build_brunel(10, seed=4)
        plan = net.compile()
        run = plan.runner(batch=2)
        run.simulate(5.0 * u.ms)
        r = run.cont(5.0 * u.ms)
        assert r[info['rec_E']].batched and run.n_traces == 1
        out = run.step()
        assert out['E']['spike'].shape == (2, 40)
        t, gids, counts = r[info['rec_E']].spike_events(trial=0)
        with pytest.raises(ValueError, match='trial'):
            r[info['rec_E']].spike_events()

    def test_poisson_stream_follows_rng_contract(self):
        net = Network(dt=DT, seed=11)
        net.add(LIF, 2)
        pg = net.add(poisson_generator, 5, params=dict(rate=20000.0 * u.Hz), name='pg')
        rec = net.observe(pg, 'spike')
        res = net.compile().runner().simulate(1.0 * u.ms)
        x = np.asarray(res[rec]['spike'])
        lam = jnp.full((5,), 2.0, jnp.float32)
        for n in (1, 5, 9):
            # K1a-6 (D-g): keys from the population name, one per id block, trial 0
            keys = step_block_keys(population_block_keys(11, 'pg', 5), n, 0)
            np.testing.assert_array_equal(x[n], poisson_counts(keys, lam, (5,)))
        assert x[0].sum() == 0  # step 0 is outside the NEST activity window

    def test_deterministic_across_runners_and_compiles(self):
        a = build_brunel(10, seed=9)
        b = build_brunel(10, seed=9)
        ra = a[0].compile().runner().simulate(30.0 * u.ms)
        rb = b[0].compile().runner().simulate(30.0 * u.ms)
        np.testing.assert_array_equal(self._spikes(ra, a[1]), self._spikes(rb, b[1]))


# --------------------------------------------------------------------------- precision

class TestPrecision:
    def test_float64(self):
        with jax.enable_x64(True):
            net, info = build_brunel(10, seed=1, record_v=True)
            plan = net.compile('float64')
            res = plan.runner().simulate(50.0 * u.ms)
            assert u.get_mantissa(res[info['rec_V']]['V_m']).dtype == jnp.float64
            assert res[info['rec_E']]['spike'].dtype == jnp.float64
            assert plan.population('E').rings['spike'].dtype == np.float64

    def test_float32_vs_float64_single_neuron(self):
        traces = []
        for prec in ('float32', 'float64'):
            with jax.enable_x64(prec == 'float64'):
                net = Network(dt=DT)
                sg = source(net, [1.0, 2.0, 2.5])
                n = net.add(iaf_psc_alpha, 1)
                net.connect(sg, n, port='ex', weight=300.0 * u.pA, delay=1.0 * u.ms)
                rec = net.observe(n, 'V_m')
                res = net.compile(prec).runner().simulate(20.0 * u.ms)
                traces.append(np.asarray(res[rec]['V_m'].to_decimal(u.mV), np.float64))
        np.testing.assert_allclose(traces[0], traces[1], atol=1e-4)

    def test_float64_needs_x64(self):
        net = Network(dt=DT)
        net.add(LIF, 1)
        with jax.enable_x64(False):
            with pytest.raises(RuntimeError, match='x64'):
                net.compile('float64')


# --------------------------------------------------------------------------- observers

class TestObservers:
    def test_intervals(self):
        net, info = build_brunel(10, seed=5, record_v=True)
        E = info['E']
        coarse = net.observe(E[:50], 'spike', 'V_m', interval=1.0 * u.ms, name='coarse')
        plan = net.compile()
        res = plan.runner().simulate(20.0 * u.ms)
        dense_s = np.asarray(res[info['rec_E']]['spike'])
        dense_v = np.asarray(res[info['rec_V']]['V_m'].to_decimal(u.mV))
        c = res[coarse]
        assert c['spike'].shape == (20, 40) and dense_s.shape == (200, 40)
        np.testing.assert_array_equal(np.asarray(c['spike']), dense_s.reshape(20, 10, 40).sum(1))
        np.testing.assert_array_equal(np.asarray(c['V_m'].to_decimal(u.mV)), dense_v[9::10])
        np.testing.assert_allclose(np.asarray(c.times.to_decimal(u.ms)), np.arange(1, 21))
        with pytest.raises(ValueError, match='multiple'):
            plan.runner().simulate(20.5 * u.ms)

    def test_recording_api(self):
        net = Network(dt=DT)
        sg = source(net, [0.2, 0.2, 0.5])
        a, b = lif(net, 1, 'A'), net.add(iaf_psc_alpha, 1, name='B')
        rec = net.observe(sg + a, 'spike')
        rv = net.observe(b, 'V_m', 'refractory_count')
        res = net.compile().runner().simulate(1.0 * u.ms)
        r = res[rec]
        assert rec in res and 'rec' not in res and len(res) == 2 and res['Observer_0'] is r
        with pytest.raises(KeyError):
            res['nope']
        with pytest.raises(KeyError, match='not recorded'):
            r['V']
        assert 'spike' in r and r.columns == {'sg': [0], 'A': [1]}
        parts = r.by_population('spike')
        assert parts['sg'].shape == (10, 1)
        t, gids, counts = r.spike_events()
        np.testing.assert_allclose(np.asarray(t.to_decimal(u.ms)), [0.2, 0.5])
        assert gids.tolist() == [0, 0] and counts.tolist() == [2.0, 1.0]
        with pytest.raises(ValueError, match='event'):
            res[rv].spike_events('V_m')
        assert res[rv]['refractory_count'].dtype == jnp.int32


# --------------------------------------------------------------------------- gradients

def test_weights_are_differentiable_through_the_program():
    net = Network(dt=DT)
    sg = source(net, [0.5, 1.0])
    n = net.add(LIF, 2, name='n')
    p = net.connect(sg, n, port='delta', weight=0.4 * u.mV, trainable=True)
    rec = net.observe(n, 'V')
    plan = net.compile()
    assert len(plan.states(brainstate.ParamState)) == 1
    prog = plan.runner().program
    params, state = prog.read_params(), prog.init_state()

    def loss(w):
        ps = {**params, 'weights': {**params['weights'], p.name: [w]}}
        _, recs = prog.rollout(ps, state, jnp.int32(0), jnp.uint32(0), 20)
        return recs[rec.name]['V'].sum()

    g = jax.grad(loss)(params['weights'][p.name][0].mantissa)
    assert np.all(np.asarray(g) > 0)


# --------------------------------------------------------------------------- K1a-5

def _readout_net(storage='dense'):
    """Poisson drive -> h (LIF, fixed) -> o (LIF) through a trainable readout ``ro``."""
    import braintools
    net = Network(dt=DT, seed=21)
    noise = net.add(poisson_generator, 12, params=dict(rate=900.0 * u.Hz), name='noise')
    h = net.add(LIF, 8, name='h', surrogate=braintools.surrogate.Sigmoid())
    o = net.add(LIF, 3, name='o', surrogate=braintools.surrogate.Sigmoid())
    net.connect(noise, h, fixed_indegree(4), port='delta', weight=0.5 * u.mV, name='nh')
    net.connect(h, o, all_to_all(), port='delta', weight=0.1 * u.mV, storage=storage,
                trainable=True, name='ro')
    rec = net.observe(o, 'V', 'spike', name='r')
    return net, rec


_TARGET = 0.6  # mV


def _delta_rule(params, state, n):
    """Readout delta rule: w += lr * outer(h spikes, target - o.V) (pure)."""
    h_spk = state['pops']['h']['spike']
    err = _TARGET - state['pops']['o']['V'].to_decimal(u.mV)
    w = params['weights']['ro'][0]
    w = w + 0.01 * jnp.outer(h_spk, err) * u.mV
    return {**params, 'weights': {**params['weights'], 'ro': [w]}}, state


class TestProgram:
    def test_hard_and_relaxed_programs_coexist(self):
        import braintools
        net = Network(dt=DT)
        net.add(LIF, 2, name='h', surrogate=braintools.surrogate.Sigmoid())
        plan = net.compile()
        hard, relaxed = Program(plan), plan.program(spikes='relaxed')
        st = hard.init_state()
        ext = {'h': {'delta': 1.0 * u.mV}}  # exactly at threshold
        f_hard = jax.jit(lambda p, s: hard.step(p, s, 0, jnp.uint32(0), ext)[1][0]['spike'])
        f_rel = jax.jit(lambda p, s: relaxed.step(p, s, 0, jnp.uint32(0), ext)[1][0]['spike'])
        p = hard.read_params()
        assert np.asarray(f_hard(p, st)).tolist() == [1.0, 1.0]
        np.testing.assert_allclose(np.asarray(f_rel(p, st)), [0.5, 0.5], rtol=1e-6)
        assert np.asarray(f_hard(p, st)).tolist() == [1.0, 1.0]  # no mode leaks back
        ra, rb = Runner(hard), Runner(relaxed)
        assert ra.program is hard and rb.program.spikes == 'relaxed' and rb.plan is plan
        ra.step(), rb.step()
        assert 'Program(spikes=' in repr(relaxed)

    def test_relaxed_rejects_models_without_smooth_spikes(self):
        from brainpy_state._kernel.models.precise import iaf_psc_exp_ps
        net = Network(dt=DT)
        net.add(LIF, 2, name='hard_lif', surrogate=None)
        with pytest.raises(ValueError, match='hard_lif'):
            net.compile().program(spikes='relaxed')
        net = Network(dt=DT)
        net.add(iaf_psc_exp_ps, 2, name='ps')
        with pytest.raises(ValueError, match='precise spike times'):
            net.compile().program(spikes='relaxed')
        net = Network(dt=DT)
        net.add(poisson_generator, 2, params=dict(rate=10.0 * u.Hz))  # devices are fine
        assert net.compile().program(spikes='relaxed').spikes == 'relaxed'
        with pytest.raises(ValueError, match='spikes'):
            net.compile().program(spikes='soft')
        with pytest.raises(TypeError, match='detach_reset'):
            net.compile().program(detach_reset=1)

    def test_replace_and_init_state(self):
        net, _ = _readout_net()
        plan = net.compile()
        prog = plan.program()
        assert prog.replace() is prog and prog.replace(spikes='hard') is prog
        rel = prog.replace(spikes='relaxed', detach_reset=True)
        assert rel.spikes == 'relaxed' and rel.detach_reset is True and rel.plan is plan
        assert rel.init_state() is prog.init_state()  # shared captured initial state
        with pytest.raises(TypeError, match='unknown Program options'):
            prog.replace(batch=2)
        b = prog.init_state(3)
        for x, y in zip(jax.tree.leaves(b), jax.tree.leaves(prog.init_state())):
            assert x.shape == (3,) + y.shape
        with pytest.raises(ValueError, match='batch'):
            prog.init_state(0)

    def test_init_state_ignores_later_model_writes(self):
        net, _ = _readout_net()
        plan = net.compile()
        prog = plan.program()
        v0 = np.asarray(prog.init_state()['pops']['h']['V'].to_decimal(u.mV)).copy()
        plan.populations[1].model.dyn_states['V'].value = 5.0 * u.mV * jnp.ones(8)
        np.testing.assert_array_equal(np.asarray(prog.init_state()['pops']['h']['V'].to_decimal(u.mV)), v0)

    def test_write_params_roundtrip(self):
        net, _ = _readout_net()
        prog = net.compile().program()
        p = prog.read_params()
        w = p['weights']['ro'][0] * 2.0
        prog.write_params({**p, 'weights': {**p['weights'], 'ro': [w]}})
        np.testing.assert_allclose(np.asarray(prog.read_params()['weights']['ro'][0].to_decimal(u.mV)),
                                   np.asarray(w.to_decimal(u.mV)))

    def test_trial_indices(self):
        assert trial_indices(None).shape == ()
        assert trial_indices(3).tolist() == [0, 1, 2]
        assert trial_indices(2, 'common').tolist() == [0, 0]
        with pytest.raises(ValueError, match='rng'):
            trial_indices(2, 'shared')
        with pytest.raises(ValueError, match='batch'):
            trial_indices(True)


class TestLearnPhase:
    @pytest.mark.parametrize('every', [1, 4])
    def test_learn_every_k_equals_updates_between_cont_calls(self, every):
        with jax.enable_x64(True):
            net, rec = _readout_net()
            plan_a = net.compile(precision='float64')
            net_b, _ = _readout_net()
            plan_b = net_b.compile(precision='float64')
            a, b = plan_a.runner(), plan_b.runner()
            # compiled: 3 + 9 steps, learn after global steps 4, 8, 12 (every=4)
            res1 = a.cont(0.3 * u.ms, learn=_delta_rule, every=every)
            res2 = a.cont(0.9 * u.ms, learn=_delta_rule, every=every)
            # reference: host-side updates between cont() calls of `every` steps
            prog_b = b.program
            V = []
            # (host loop over update windows: the reference the compiled learn phase must match)
            for _ in range(12 // every):
                r = b.cont(every * DT)
                V.append(np.asarray(r[rec]['V'].to_decimal(u.mV)))
                p, _ = _delta_rule(prog_b.read_params(), b.state, b.n)
                prog_b.write_params(p)
            V_a = np.concatenate([np.asarray(res1[rec]['V'].to_decimal(u.mV)),
                                  np.asarray(res2[rec]['V'].to_decimal(u.mV))])
            np.testing.assert_allclose(V_a, np.concatenate(V), rtol=1e-12, atol=1e-12)
            wa = np.asarray(plan_a.projection('ro').groups[0].weight.value.to_decimal(u.mV))
            wb = np.asarray(plan_b.projection('ro').groups[0].weight.value.to_decimal(u.mV))
            np.testing.assert_allclose(wa, wb, rtol=1e-12)
            assert not np.allclose(wa, 0.1)  # the readout learned
            assert a.n == 12 and a.n_traces == 2

    def test_rollout_returns_params_only_with_learn(self):
        net, rec = _readout_net()
        prog = net.compile().program()
        p, s = prog.read_params(), prog.init_state()
        out = prog.rollout(p, s, 0, jnp.uint32(0), 5)
        assert len(out) == 2
        st, recs, p2 = prog.rollout(p, s, 0, jnp.uint32(0), 5, learn=_delta_rule, every=2)
        assert jax.tree.structure(p2) == jax.tree.structure(p)
        assert not np.allclose(np.asarray(u.get_mantissa(p2['weights']['ro'][0])),
                               np.asarray(u.get_mantissa(p['weights']['ro'][0])))

    def test_learn_can_reset_state(self):
        """A trial reset in the learn phase (state change), every 5 steps."""
        net, rec = _readout_net()
        prog = net.compile().program()
        init = prog.init_state()

        def reset(params, state, n):
            return params, {**state, 'pops': init['pops']}

        st, recs, _ = prog.rollout(prog.read_params(), init, 0, jnp.uint32(0), 10, learn=reset,
                                   every=5)
        for x, y in zip(jax.tree.leaves(st['pops']), jax.tree.leaves(init['pops'])):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

    def test_learn_with_a_stateful_optimizer(self):
        """An optimizer step every k steps: braintools optimizer states are carried by the scan."""
        import braintools
        net, rec = _readout_net()
        plan = net.compile()
        prog = plan.program()
        ro = plan.projection('ro').groups[0].weight
        opt = braintools.optim.Adam(lr=1e-3)
        opt.register_trainable_weights({'ro': ro})

        def learn(params, state, n):
            h_spk = state['pops']['h']['spike']
            err = _TARGET - state['pops']['o']['V'].to_decimal(u.mV)
            opt.update({'ro': -jnp.outer(h_spk, err) * u.mV})
            return prog.read_params(), state

        run = Runner(prog)
        w0 = np.asarray(ro.value.to_decimal(u.mV)).copy()
        run.cont(2.0 * u.ms, learn=learn, every=5)
        w1 = np.asarray(ro.value.to_decimal(u.mV))
        assert not np.allclose(w1, w0) and np.all(np.isfinite(w1))

    def test_learn_errors(self):
        net, rec = _readout_net()
        plan = net.compile()
        prog = plan.program()
        p, s = prog.read_params(), prog.init_state()
        with pytest.raises(TypeError, match='callable'):
            prog.rollout(p, s, 0, jnp.uint32(0), 3, learn=3)
        with pytest.raises(ValueError, match='every'):
            prog.rollout(p, s, 0, jnp.uint32(0), 3, learn=_delta_rule, every=0)
        with pytest.raises(ValueError, match='every'):
            prog.rollout(p, s, 0, jnp.uint32(0), 3, learn=_delta_rule, every=True)
        with pytest.raises(TypeError, match='structure'):
            prog.rollout(p, s, 0, jnp.uint32(0), 3, learn=lambda a, b, n: ({}, b))
        with pytest.raises(NotImplementedError, match='batched'):
            plan.runner(batch=2).cont(1.0 * u.ms, learn=_delta_rule)
