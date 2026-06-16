# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``astrocyte_brunel_{bernoulli,fixed_indegree}`` (NEST demos, goal 24).

Ports of NEST's two Brunel-with-astrocytes demos: a balanced random network of
``aeif_cond_alpha_astro`` neurons (excitatory + inhibitory) plus ``astrocyte_lr_1994``
astrocytes, where the excitatory population projects to both neurons and astrocytes
through ``tripartite_connect`` with the ``third_factor_bernoulli_with_pool`` rule
(``pool_size=10``, ``pool_type='random'``). The two ports differ only in the primary
neuron->neuron rule (``pairwise_bernoulli`` vs ``fixed_indegree``).

Validation strategy. Unlike the deterministic ``astrocyte_small_network``, the Brunel
ports are validated **distributionally on connectivity**, not on firing rate:

* The astrocyte-pool connectivity rule (what the 17b placeholder was blocked on) is
  the new surface; its ``pre->post`` / ``pre->astro`` / ``astro->post`` realized
  edge counts are compared to live NEST seed-by-seed (category D, 5 %), for **both**
  primary rules. This is robust (pure connectivity, no dynamics chaos) and directly
  exercises ``random`` pools at the demos' parameters.
* The per-edge SIC dynamics are already validated tightly against NEST elsewhere
  (the ``tripartite_connect`` micro-parity GATE ``tripartite_connect_test.py`` and the
  deterministic ``astrocyte_small_network_test.py``); they are not re-compared here.
* Firing-rate parity is **not** asserted: a balanced asynchronous-irregular Brunel
  state needs near-full scale (the demos' ``p=0.1`` gives ~800 inhibitory inputs per
  neuron only at ``N=10^4``); at the dense-friendly test scale the strong 2 kHz drive
  is not balanced, so a rate match would be neither meaningful nor stable. The
  NEST-free law instead asserts the network lowers, stays finite, fires and closes
  the SIC loop.

Synapse divergence (goal 24 spec §3): both sides use ``static_synapse`` for the
primary/third_in arms (NEST's demo uses ``tsodyks_synapse``); the connectivity is
identical, and the parity here is on connectivity only.
"""
import gc
import unittest

import brainstate
import jax
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

try:
    import nest
except Exception:                                         # pragma: no cover - env dependent
    nest = None

from brainpy_state import pairwise_bernoulli, fixed_indegree
from brainpy_state._nest_validation.nest_compare import requires_nest, compare_distributional
from brainpy_state._nest_validation.tolerance_conventions import CAT_D

from examples.nest.astrocyte_brunel_bernoulli import build, run as run_bernoulli
from examples.nest.astrocyte_brunel_fixed_indegree import run as run_fixed_indegree

DT = 0.1

#: Connectivity-parity config (small + dense-friendly; connectivity only, no sim).
CONN = dict(n_ex=100, n_in=25, n_astro=80, p_primary=0.2, ce=20, ci=5, p_third=0.5,
            pool_size=10, pool_type='random')
CONN_SEEDS = (1, 2, 3, 4)

#: Build kwargs shared by every ``build`` call (weights / delays / params are the
#: demo's; only the primary rule and seed vary).
_BUILD_KW = dict(poisson_rate=2000.0, w_e=1.0, w_i=4.0, w_a2n=0.05, d_e=2.0, d_i=1.0,
                 tau_syn_ex=2.0, tau_syn_in=4.0, IP3_0=0.4, sic_delay_steps=10)


def _edges(proj):
    """Unique (source, target) population-local pairs of a realized projection."""
    if proj is None:
        return set()
    pe = proj.realized_edges()
    s = np.asarray(pe.source); t = np.asarray(pe.target)
    return set(zip(s.tolist(), t.tolist()))


def _pool_sizes_by_target(a2n_edges):
    """Map each target -> number of distinct source astrocytes feeding it (a2n)."""
    by_t = {}
    for s, t in a2n_edges:
        by_t.setdefault(t, set()).add(s)
    return {t: len(srcs) for t, srcs in by_t.items()}


def _primary_rule(kind, cfg):
    """The excitatory/inhibitory primary rule factory for a port ``kind``."""
    if kind == 'bernoulli':
        return lambda _role: pairwise_bernoulli(cfg['p_primary'])
    return lambda role: fixed_indegree(cfg['ce'] if role == 'ex' else cfg['ci'])


def _bp_brunel_edges(kind, seed, cfg):
    """brainpy realized edge sets for a Brunel-astro port (connectivity only)."""
    from brainpy_state import Simulator
    brainstate.random.seed(seed)
    sim = Simulator(dt=DT * u.ms)
    net = build(sim, primary_rule=_primary_rule(kind, cfg), n_ex=cfg['n_ex'],
                n_in=cfg['n_in'], n_astro=cfg['n_astro'], p_third=cfg['p_third'],
                pool_size=cfg['pool_size'], pool_type=cfg['pool_type'], seed=seed,
                **_BUILD_KW)
    return {'n2n': _edges(net['primary']), 'n2a': _edges(net['third_in']),
            'a2n': _edges(net['third_out']), 'inh': _edges(net['inhibitory'])}


def _nest_edges(conn, src0, tgt0):
    """Live-NEST GetConnections -> a set of 0-indexed (source, target) local pairs."""
    d = conn.get(['source', 'target'])
    s = np.atleast_1d(np.asarray(d['source']))
    t = np.atleast_1d(np.asarray(d['target']))
    if s.size == 0:
        return set()
    return set(zip((s - src0).tolist(), (t - tgt0).tolist()))


def _nest_brunel_edges(kind, seed, cfg):
    """Live-NEST realized edge sets for the same Brunel-astro port."""
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': DT, 'rng_seed': int(seed)})
    N = cfg['n_ex'] + cfg['n_in']
    neurons = nest.Create('aeif_cond_alpha_astro', N,
                          params={'tau_syn_ex': 2.0, 'tau_syn_in': 4.0})
    astro = nest.Create('astrocyte_lr_1994', cfg['n_astro'], params={'IP3': 0.4})
    ex, inh = neurons[:cfg['n_ex']], neurons[cfg['n_ex']:]
    if kind == 'bernoulli':
        conn_e = {'rule': 'pairwise_bernoulli', 'p': cfg['p_primary']}
        conn_i = {'rule': 'pairwise_bernoulli', 'p': cfg['p_primary']}
    else:
        conn_e = {'rule': 'fixed_indegree', 'indegree': cfg['ce']}
        conn_i = {'rule': 'fixed_indegree', 'indegree': cfg['ci']}
    nest.TripartiteConnect(
        ex, neurons, astro, conn_spec=conn_e,
        third_factor_conn_spec={'rule': 'third_factor_bernoulli_with_pool',
                                'p': cfg['p_third'], 'pool_size': cfg['pool_size'],
                                'pool_type': cfg['pool_type']},
        syn_specs={'primary': {'synapse_model': 'static_synapse'},
                   'third_in': {'synapse_model': 'static_synapse'},
                   'third_out': {'synapse_model': 'sic_connection'}})
    nest.Connect(inh, neurons, conn_i, syn_spec={'synapse_model': 'static_synapse', 'weight': -4.0})
    n0, a0 = neurons.tolist()[0], astro.tolist()[0]
    return {'n2n': _nest_edges(nest.GetConnections(ex, neurons), n0, n0),
            'n2a': _nest_edges(nest.GetConnections(ex, astro), n0, a0),
            'a2n': _nest_edges(nest.GetConnections(astro, neurons), a0, n0),
            'inh': _nest_edges(nest.GetConnections(inh, neurons), n0, n0)}


class TestAstrocyteBrunelLaw(unittest.TestCase):
    """Structural / dynamical invariants that need no NEST (always run)."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def _assert_runs_and_couples(self, out):
        ip3 = np.asarray(u.get_mantissa(out['IP3']))
        isic = np.asarray(u.get_mantissa(out['I_SIC']))
        self.assertTrue(np.all(np.isfinite(ip3)) and np.all(np.isfinite(isic)))
        self.assertGreater(out['e_rate'], 0.0, 'excitatory neurons fire')
        self.assertGreater(out['i_rate'], 0.0, 'inhibitory neurons fire')
        self.assertGreater(float(ip3.max()), 0.4, 'the pre->astro arm raises IP3')
        # The astrocyte Ca integrator is slow: the SIC loop ignites only after a
        # ~250-300 ms latency at these drives, so the window must clear it.
        self.assertGreater(float(isic.max()), 0.0, 'the SIC arm delivers current')

    def test_bernoulli_lowers_and_couples(self):
        """The Bernoulli port lowers under for_loop, fires, and closes the SIC loop."""
        self._assert_runs_and_couples(
            run_bernoulli(sim_time=400.0, n_ex=80, n_in=20, n_astro=100, seed=1))

    def test_fixed_indegree_lowers_and_couples(self):
        """The fixed-indegree port lowers under for_loop, fires, and closes the SIC loop."""
        self._assert_runs_and_couples(
            run_fixed_indegree(sim_time=400.0, n_ex=80, n_in=20, n_astro=100, seed=1))

    def test_random_pool_invariant_both_rules(self):
        """Each target's distinct astrocytes never exceed pool_size (both ports)."""
        for kind in ('bernoulli', 'fixed_indegree'):
            a2n = _bp_brunel_edges(kind, seed=1, cfg=CONN)['a2n']
            sizes = _pool_sizes_by_target(a2n)
            self.assertTrue(sizes, f'{kind}: some targets receive astrocyte input')
            self.assertTrue(all(v <= CONN['pool_size'] for v in sizes.values()),
                            f'{kind}: distinct astrocytes per target must be <= pool_size')


@requires_nest
class TestAstrocyteBrunelConnectivityParity(unittest.TestCase):
    """Seed-mean realized edge counts match live NEST (category D), both primary rules."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def _parity_for(self, kind):
        bp_counts = {k: [] for k in ('n2n', 'n2a', 'a2n', 'inh')}
        nest_counts = {k: [] for k in ('n2n', 'n2a', 'a2n', 'inh')}
        for s in CONN_SEEDS:
            b = _bp_brunel_edges(kind, s, CONN)
            n = _nest_brunel_edges(kind, s, CONN)
            for k in bp_counts:
                bp_counts[k].append(len(b[k]))
                nest_counts[k].append(len(n[k]))
            # The hard pool invariant must hold on the NEST side too.
            sizes = _pool_sizes_by_target(n['a2n'])
            self.assertTrue(all(v <= CONN['pool_size'] for v in sizes.values()),
                            f'{kind}: NEST distinct astrocytes per target <= pool_size')
        for k in ('n2n', 'n2a', 'a2n', 'inh'):
            self.assertGreater(float(np.mean(nest_counts[k])), 0.0)
            compare_distributional(nest_counts[k], bp_counts[k], tol=CAT_D,
                                   metric=f'{kind} {k} edge count').assert_()

    def test_bernoulli_edge_counts_match_nest(self):
        """Bernoulli-primary n2n/n2a/a2n/inh seed-mean counts land within CAT_D of NEST."""
        self._parity_for('bernoulli')

    def test_fixed_indegree_edge_counts_match_nest(self):
        """Fixed-indegree-primary n2n/n2a/a2n/inh seed-mean counts land within CAT_D of NEST."""
        self._parity_for('fixed_indegree')


if __name__ == '__main__':
    unittest.main()
