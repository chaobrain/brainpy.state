# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""NEST-free tests for connection enumeration + introspection.

Exercises the per-projection ``realized_edges()`` accessor (Phase 2) and the
``Simulator.get_connections(...) -> SynapseCollection`` API (Phase 3): edge order,
source/target/synapse filtering, ``get``/``set`` round-trips, the homogeneous /
plastic ``set`` guards, and the post-simulation live-weight read.
"""
import jax
import numpy as np
import pytest

jax.config.update('jax_enable_x64', True)

import brainstate
import brainunit as u

brainstate.environ.set(precision=64, platform='cpu')

from brainpy_state import (
    Simulator, iaf_psc_exp, iaf_cond_exp, all_to_all, one_to_one, fixed_indegree,
    static_synapse, static_synapse_hom_w,
)
from brainpy_state._nest_network._event_plastic import _StaticTestRule


def _no_autapse_pairs(n):
    return [(i, j) for i in range(n) for j in range(n) if i != j]


class evolving_synapse(_StaticTestRule):
    """Test-only plastic rule whose model name is NOT in the static family.

    Used to exercise the ``.set('weight')`` guard: a rule outside the
    ``static_synapse`` allowlist is treated as weight-evolving (rule-managed), so a
    write-back is refused regardless of whether this constant kernel evolves.
    """


class _RampRule(_StaticTestRule):
    """Test-only plastic rule that ramps every edge weight by +1 (mantissa) per step.

    Lets a test observe the live post-simulation weight read (evolved ≠ init).
    """

    def update(self, state, ctx):
        w = state['weight'] + 1.0
        return {'weight': w}, w


# ----------------------------------------------------------------------------
# Phase 2 — per-projection realized_edges()
# ----------------------------------------------------------------------------

class TestRealizedEdgesStatic:
    def test_dense_all_to_all_no_autapse_enumeration(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 3)
        proj = sim.connect(pop, pop, rule=all_to_all, weight=20. * u.pA,
                           allow_autapses=False, comm='dense')
        e = proj.realized_edges()
        got = sorted(zip(np.asarray(e.source).tolist(),
                         np.asarray(e.target).tolist()))
        assert got == _no_autapse_pairs(3)
        assert e.is_homogeneous_weight is False     # dense holds a per-edge matrix
        assert e.is_plastic is False
        assert e.model_name == 'static_synapse'

    def test_dense_weight_matches_matrix(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 4)
        proj = sim.connect(pop, pop, rule=all_to_all, weight=7. * u.pA,
                           allow_autapses=False, comm='dense')
        e = proj.realized_edges()
        W = u.get_mantissa(proj._W.value)          # (n_pre, n_post) dense matrix
        w = u.Quantity(e.weight).to_decimal(u.pA)
        src = np.asarray(e.source); trg = np.asarray(e.target)
        assert np.allclose(w, W[src, trg])
        assert np.allclose(w, 7.0)

    def test_sparse_matches_dense_enumeration(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 5)
        proj = sim.connect(pop, pop, rule=all_to_all, weight=3. * u.pA,
                           allow_autapses=False, comm='sparse')
        e = proj.realized_edges()
        got = sorted(zip(np.asarray(e.source).tolist(),
                         np.asarray(e.target).tolist()))
        assert got == _no_autapse_pairs(5)
        assert e.is_homogeneous_weight is False     # sparse holds per-edge _data

    def test_one_to_one_is_homogeneous(self):
        sim = Simulator(dt=0.1 * u.ms)
        a = sim.create(iaf_psc_exp, 4)
        b = sim.create(iaf_psc_exp, 4)
        proj = sim.connect(a, b, rule=one_to_one, weight=5. * u.pA)
        e = proj.realized_edges()
        assert np.asarray(e.source).tolist() == [0, 1, 2, 3]
        assert np.asarray(e.target).tolist() == [0, 1, 2, 3]
        assert e.is_homogeneous_weight is True      # shared scalar weight
        assert e.is_plastic is False


class TestRealizedEdgesPlastic:
    def test_plastic_static_synapse_per_edge(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 3)
        proj = sim.connect(pop, pop, rule=all_to_all,
                           synapse=static_synapse(weight=11. * u.pA),
                           allow_autapses=False)
        e = proj.realized_edges()
        got = sorted(zip(np.asarray(e.source).tolist(),
                         np.asarray(e.target).tolist()))
        assert got == _no_autapse_pairs(3)
        assert e.is_plastic is True
        assert e.is_homogeneous_weight is False
        assert e.model_name == 'static_synapse'
        # weight readable BEFORE simulate (from the init array, not the State)
        w = u.Quantity(e.weight).to_decimal(u.pA)
        assert np.allclose(w, 11.0)

    def test_plastic_hom_w_is_homogeneous(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 3)
        proj = sim.connect(pop, pop, rule=all_to_all,
                           synapse=static_synapse_hom_w(weight=9. * u.pA),
                           allow_autapses=False)
        e = proj.realized_edges()
        assert e.is_homogeneous_weight is True
        assert e.is_plastic is True
        w = u.Quantity(e.weight).to_decimal(u.pA)
        assert np.allclose(w, 9.0)                  # 0-d State broadcasts to E
        assert u.get_mantissa(e.weight).shape == (e.source.shape[0],)


class TestRealizedEdgesMultapses:
    def test_sparse_enumerates_multapses_distinctly(self):
        sim = Simulator(dt=0.1 * u.ms)
        pre = sim.create(iaf_psc_exp, 4)
        post = sim.create(iaf_psc_exp, 3)
        # K=4 indegree over 3 post = 12 directed edges; allow_multapses lets a
        # (pre,post) pair repeat, and sparse CSR stores each repeat distinctly.
        proj = sim.connect(pre, post, rule=fixed_indegree(4), weight=1. * u.pA,
                           comm='sparse', allow_multapses=True, seed=1)
        e = proj.realized_edges()
        assert e.source.shape[0] == 12              # every directed edge, repeats kept


# ----------------------------------------------------------------------------
# Phase 3 — Simulator.get_connections + SynapseCollection
# ----------------------------------------------------------------------------

class TestGetConnectionsEnumeration:
    def test_single_projection_all_edges(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 4)
        sim.connect(pop, pop, rule=all_to_all, weight=2. * u.pA,
                    allow_autapses=False, comm='sparse')
        conns = sim.get_connections()
        assert len(conns) == 12                     # 4 * 3 no-autapse
        w = u.Quantity(conns.get('weight')).to_decimal(u.pA)
        assert np.allclose(w, 2.0)
        assert np.asarray(conns.source).shape[0] == 12

    def test_source_target_direction_filter(self):
        sim = Simulator(dt=0.1 * u.ms)
        e = sim.create(iaf_psc_exp, 3)
        i = sim.create(iaf_psc_exp, 2)
        sim.connect(e, i, rule=all_to_all, weight=1. * u.pA, comm='sparse')
        sim.connect(i, e, rule=all_to_all, weight=-1. * u.pA, comm='sparse')
        ei = sim.get_connections(source=e, target=i)
        assert len(ei) == 6
        assert np.allclose(u.Quantity(ei.get('weight')).to_decimal(u.pA), 1.0)
        ie = sim.get_connections(source=i, target=e)
        assert len(ie) == 6
        assert np.allclose(u.Quantity(ie.get('weight')).to_decimal(u.pA), -1.0)

    def test_empty_result(self):
        sim = Simulator(dt=0.1 * u.ms)
        e = sim.create(iaf_psc_exp, 3)
        i = sim.create(iaf_psc_exp, 2)
        sim.connect(e, i, rule=all_to_all, weight=1. * u.pA, comm='sparse')
        conns = sim.get_connections(source=i, target=e)   # no i->e edges exist
        assert len(conns) == 0
        assert np.asarray(conns.source).shape[0] == 0
        assert u.get_mantissa(conns.get('weight')).shape[0] == 0

    def test_sliced_source_view(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 5)
        sim.connect(pop, pop, rule=all_to_all, weight=1. * u.pA,
                    allow_autapses=False, comm='sparse')
        conns = sim.get_connections(source=pop[:2])        # sources {0, 1} only
        assert set(np.asarray(conns.source).tolist()) <= {0, 1}
        assert len(conns) == 2 * 4                         # each of 2 sources -> 4 targets

    def test_repr_and_iter(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 3)
        sim.connect(pop, pop, rule=all_to_all, weight=1. * u.pA,
                    allow_autapses=False, comm='sparse')
        conns = sim.get_connections()
        assert 'SynapseCollection' in repr(conns)
        pairs = list(conns)
        assert len(pairs) == len(conns)
        assert all(len(p) == 2 for p in pairs)             # (source, target) per edge


class TestGetConnectionsGetSet:
    def test_get_multiple_keys_returns_dict(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 3)
        sim.connect(pop, pop, rule=all_to_all, weight=4. * u.pA,
                    allow_autapses=False, comm='sparse')
        conns = sim.get_connections()
        d = conns.get(['source', 'target', 'weight'])
        assert set(d.keys()) == {'source', 'target', 'weight'}
        assert np.asarray(d['source']).shape[0] == 6
        assert np.allclose(u.Quantity(d['weight']).to_decimal(u.pA), 4.0)

    def test_set_weight_array_roundtrip(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 3)
        sim.connect(pop, pop, rule=all_to_all, weight=1. * u.pA,
                    allow_autapses=False, comm='sparse')
        conns = sim.get_connections()
        n = len(conns)
        conns.set('weight', (np.arange(n) + 10.0) * u.pA)
        got = u.Quantity(conns.get('weight')).to_decimal(u.pA)
        assert np.allclose(got, np.arange(n) + 10.0)

    def test_set_weight_scalar_broadcast(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 3)
        sim.connect(pop, pop, rule=all_to_all, weight=1. * u.pA,
                    allow_autapses=False, comm='dense')
        conns = sim.get_connections()
        conns.set('weight', 7. * u.pA)
        assert np.allclose(u.Quantity(conns.get('weight')).to_decimal(u.pA), 7.0)

    def test_delay_none_reads_zero(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 3)
        sim.connect(pop, pop, rule=all_to_all, weight=1. * u.pA,
                    allow_autapses=False, comm='sparse')   # delay=None
        conns = sim.get_connections()
        d = u.Quantity(conns.get('delay')).to_decimal(u.ms)
        assert np.allclose(d, 0.0)

    def test_set_delay_grid_rounded(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 2)
        sim.connect(pop, pop, rule=all_to_all, weight=1. * u.pA, delay=1.0 * u.ms,
                    allow_autapses=False, comm='sparse')
        conns = sim.get_connections()
        conns.set('delay', 0.17 * u.ms)                    # round(1.7) = 2 steps -> 0.2 ms
        d = u.Quantity(conns.get('delay')).to_decimal(u.ms)
        assert np.allclose(d, 0.2)


class TestGetConnectionsGuards:
    def test_homogeneous_per_edge_set_refused(self):
        sim = Simulator(dt=0.1 * u.ms)
        a = sim.create(iaf_psc_exp, 3)
        b = sim.create(iaf_psc_exp, 3)
        sim.connect(a, b, rule=one_to_one, weight=5. * u.pA)
        conns = sim.get_connections()
        with pytest.raises(ValueError):
            conns.set('weight', np.arange(3) * u.pA)       # per-edge on homogeneous

    def test_homogeneous_scalar_set_allowed(self):
        sim = Simulator(dt=0.1 * u.ms)
        a = sim.create(iaf_psc_exp, 3)
        b = sim.create(iaf_psc_exp, 3)
        sim.connect(a, b, rule=one_to_one, weight=5. * u.pA)
        conns = sim.get_connections()
        conns.set('weight', 8. * u.pA)                     # scalar broadcast is fine
        assert np.allclose(u.Quantity(conns.get('weight')).to_decimal(u.pA), 8.0)

    def test_synapse_model_filter_and_evolving_guard(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 3)
        sim.connect(pop, pop, rule=all_to_all, weight=1. * u.pA,
                    allow_autapses=False, comm='sparse')                 # static
        sim.connect(pop, pop, synapse=evolving_synapse(weight=2. * u.pA),
                    allow_autapses=False)                                # plastic, evolving
        static = sim.get_connections(synapse='static_synapse')
        evolv = sim.get_connections(synapse='evolving_synapse')
        assert len(static) == 6
        assert len(evolv) == 6
        assert np.allclose(u.Quantity(static.get('weight')).to_decimal(u.pA), 1.0)
        assert np.allclose(u.Quantity(evolv.get('weight')).to_decimal(u.pA), 2.0)
        with pytest.raises(ValueError):
            evolv.set('weight', 5. * u.pA)                 # rule-managed -> refused

    def test_plastic_static_synapse_is_settable(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 3)
        sim.connect(pop, pop, synapse=static_synapse(weight=1. * u.pA),
                    allow_autapses=False)
        conns = sim.get_connections()
        n = len(conns)
        conns.set('weight', (np.arange(n) + 3.0) * u.pA)   # static family -> settable
        assert np.allclose(u.Quantity(conns.get('weight')).to_decimal(u.pA),
                           np.arange(n) + 3.0)


class TestGetConnectionsLiveRead:
    def test_plastic_weight_live_read_after_simulate(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 2)
        sim.connect(pop, pop, synapse=_RampRule(weight=1. * u.pA),
                    allow_autapses=False)
        conns = sim.get_connections()
        w0 = u.Quantity(conns.get('weight')).to_decimal(u.pA)
        assert np.allclose(w0, 1.0)                        # pre-simulate init
        sim.simulate(1.0 * u.ms)                           # 10 steps of +1 each
        w1 = u.Quantity(conns.get('weight')).to_decimal(u.pA)
        assert np.all(w1 > w0)                             # live-read reflects evolution
        assert np.allclose(w1, 11.0)

    def test_plastic_weight_set_after_simulate_writes_live_state(self):
        # Once simulate() has allocated the weight State, set('weight') must write the
        # *live* State (not the pre-sim _w_init array) so the edit is observable.
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 2)
        sim.connect(pop, pop, synapse=static_synapse(weight=1. * u.pA),
                    allow_autapses=False)                  # static -> non-evolving, settable
        sim.simulate(0.5 * u.ms)                           # init_state -> live weight State
        conns = sim.get_connections()
        n = len(conns)
        conns.set('weight', (np.arange(n) + 2.0) * u.pA)   # writes the live State
        assert np.allclose(u.Quantity(conns.get('weight')).to_decimal(u.pA),
                           np.arange(n) + 2.0)


class TestGetConnectionsEdgeCases:
    """Error guards, the delay->0 seam clear, the receptor comm mode, empty reads."""

    def test_get_unknown_key_raises(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 2)
        sim.connect(pop, pop, rule=all_to_all, weight=1. * u.pA,
                    allow_autapses=False, comm='sparse')
        conns = sim.get_connections()
        with pytest.raises(KeyError):
            conns.get('nonsense')                          # not source/target/weight/delay

    def test_set_unknown_key_raises(self):
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 2)
        sim.connect(pop, pop, rule=all_to_all, weight=1. * u.pA,
                    allow_autapses=False, comm='sparse')
        conns = sim.get_connections()
        with pytest.raises(KeyError):
            conns.set('source', np.zeros(len(conns)))      # source/target are read-only

    def test_set_delay_on_plastic_refused(self):
        # A plastic projection's delay is fixed at connect(); SynapseCollection won't set it.
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 3)
        sim.connect(pop, pop, synapse=static_synapse(weight=1. * u.pA),
                    allow_autapses=False)
        conns = sim.get_connections()
        with pytest.raises(ValueError):
            conns.set('delay', 0.3 * u.ms)

    def test_set_delay_rounds_to_zero_clears_seam(self):
        # A sub-half-grid delay rounds to 0 steps: the InputDelay seam is cleared and the
        # read-back delay is 0 ms (NEST stores delays as integer multiples of resolution).
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 2)
        sim.connect(pop, pop, rule=all_to_all, weight=1. * u.pA, delay=1.0 * u.ms,
                    allow_autapses=False, comm='sparse')
        conns = sim.get_connections()
        conns.set('delay', 0.04 * u.ms)                    # round(0.4) = 0 steps -> 0 ms
        d = u.Quantity(conns.get('delay')).to_decimal(u.ms)
        assert np.allclose(d, 0.0)

    def test_receptor_projection_enumerates_and_sets(self):
        # The fourth comm mode: a receptor-routed EventProjection (per-receptor scatter).
        sim = Simulator(dt=0.1 * u.ms)
        pre = sim.create(iaf_psc_exp, 3)
        post = sim.create(iaf_cond_exp, 2)                 # multi-receptor (g_ex / g_in)
        sim.connect(pre, post, rule=all_to_all, receptor_type=1, weight=1.5 * u.nS)
        conns = sim.get_connections()
        assert len(conns) == 6                             # 3 x 2 all-to-all
        assert np.allclose(u.Quantity(conns.get('weight')).to_decimal(u.nS), 1.5)
        conns.set('weight', (np.arange(6) + 1.0) * u.nS)   # per-edge (non-homogeneous)
        assert np.allclose(u.Quantity(conns.get('weight')).to_decimal(u.nS),
                           np.arange(6) + 1.0)

    def test_empty_collection_source_and_target(self):
        sim = Simulator(dt=0.1 * u.ms)
        e = sim.create(iaf_psc_exp, 3)
        i = sim.create(iaf_psc_exp, 2)
        sim.connect(e, i, rule=all_to_all, weight=1. * u.pA, comm='sparse')
        conns = sim.get_connections(source=i, target=e)    # no i->e edges exist
        assert conns.source.shape[0] == 0
        assert conns.target.shape[0] == 0

    def test_homogeneous_plastic_scalar_set_pre_and_post_simulate(self):
        # static_synapse_hom_w shares one scalar weight (it is in the settable static
        # family); a scalar set writes that shared value both before init_state (the
        # _w_init scalar path) and after simulate (the live 0-d weight State).
        sim = Simulator(dt=0.1 * u.ms)
        pop = sim.create(iaf_psc_exp, 3)
        sim.connect(pop, pop, synapse=static_synapse_hom_w(weight=1. * u.pA),
                    allow_autapses=False)
        conns = sim.get_connections()
        conns.set('weight', 4. * u.pA)                     # pre-sim: _w_init scalar branch
        assert np.allclose(u.Quantity(conns.get('weight')).to_decimal(u.pA), 4.0)
        sim.simulate(0.3 * u.ms)                           # allocate the live weight State
        conns2 = sim.get_connections()
        conns2.set('weight', 6. * u.pA)                    # post-sim: live 0-d State branch
        assert np.allclose(u.Quantity(conns2.get('weight')).to_decimal(u.pA), 6.0)
