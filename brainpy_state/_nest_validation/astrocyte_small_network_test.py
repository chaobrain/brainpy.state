# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity for ``astrocyte_small_network`` (NEST tripartite demo, goal 24).

Port of NEST's ``astrocyte_small_network``: 10 presynaptic + 10 postsynaptic
``aeif_cond_alpha_astro`` neurons and 5 ``astrocyte_lr_1994`` astrocytes, wired with
``tripartite_connect`` (primary ``pairwise_bernoulli`` ``p=1``; third factor
``third_factor_bernoulli_with_pool`` ``p=1``, ``pool_size=1``, ``pool_type='block'``).
The whole network is deterministic (current-driven, no PRNG), so it is compared to
live NEST **per-sample** (trace parity), not distributionally.

Two pillars (each pairs a NEST-free law that always runs with a ``@requires_nest``
parity):

* **Structural / dynamical law** -- the tripartite net lowers under the Simulator's
  ``for_loop``, every trace is finite, IP3 climbs with the presynaptic drive and the
  SIC arm delivers a nonzero ``I_SIC`` back to the postsynaptic neurons.
* **Loop trace parity** -- the deterministic driver ``V_pre`` (``CAT_A``), the
  astrocytic IP3/Ca (``ASTRO_TOL``) and the astrocyte->neuron output ``I_SIC``
  (``SIC_TOL``) track live NEST sample-for-sample. The downstream ``V_post`` is
  **not** sample-wise compared: the postsynaptic neurons sit in a strongly-shunted,
  near-critical spiking regime (a 10 nS simultaneous EPSP conductance plus ``I_e``),
  so their membrane potential and exact spike count are chaotically sensitive to
  sub-step timing (a flipped spike step is a ~25 mV transient). This mirrors the 15d
  convention, which validated the loop coupling (IP3->Ca->I_SIC) and the ``I_e``
  driver, never a synaptically-driven postsynaptic ``V``. ``I_SIC`` rides a slightly
  relaxed tolerance because the dense-merge multiplies the SIC log-onset singularity
  (``d SIC/d Ca -> inf`` as Ca approaches threshold) by the merged weight ``N``.

The block symmetry (all-to-all primary, ``pool_size=1`` block) makes every pre
neuron, every astrocyte and every post neuron receive identical input, so the
network is the 15d single SIC loop replicated; the brainpy dense-merge sums the
duplicate (multigraph) edges into one edge of summed weight, delivering exactly
NEST's per-edge drive (``N`` connections of weight ``w`` == one merged edge of
weight ``N·w``). Synapses are static on both sides (goal 24 spec §3): NEST's demo
uses ``tsodyks_synapse`` for the primary/third_in arms, but the parity harness pins
both sides to ``static_synapse`` + ``sic_connection`` for an apples-to-apples match.

Realized aligned ``max|Δ|`` (measured, recorded in the cluster spec §8): see the
assertions below; IP3/Ca/I_SIC ride ``ASTRO_TOL`` and the mean V rides ``CAT_A``.
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

from brainpy_state._nest_validation.nest_compare import requires_nest, compare_trace
from brainpy_state._nest_validation.tolerance_conventions import TraceTolerance, CAT_A

from examples.nest.astrocyte_small_network import run

DT = 0.1
SIC_DELAY_STEPS = 10

#: Astrocyte state trace tolerance (IP3/Ca µM); align_steps=3 absorbs the
#: spike->IP3 + sic-delivery integer pipeline offset (15d).
ASTRO_TOL = TraceTolerance(1e-3, 1e-3, align_steps=3, label='A',
                           note='astrocyte IP3/Ca vs live NEST (one-step pipeline align)')

#: Delivered-current (I_SIC) tolerance. The dense-merge collapses the N duplicate
#: astro->post edges into one of weight N·w, multiplying the SIC log-onset
#: singularity (d SIC/d Ca -> inf as Ca -> threshold) by N; the threshold-crossing
#: transient lands at ~0.7 % relative here (Ca itself matches to ~6e-5), so I_SIC
#: rides a 2 % relative bound (still ~3x the measured max|Δ|).
SIC_TOL = TraceTolerance(1e-3, 2e-2, align_steps=3, label='C',
                         note='delivered I_SIC vs live NEST (merged-weight log-onset)')

#: Deterministic parity config (short window keeps it fast; demo runs 1000 ms).
DET = dict(sim_time=300.0, dt=DT, n_neurons=10, n_astro=5, p_primary=1.0, p_third=1.0,
           pool_size=1, pool_type='block', I_e=1000.0, tau_syn_ex=2.0, delta_IP3=0.2,
           IP3_0=0.4, w_primary=1.0, w_third_in=1.0, w_a2n=1.0, conn_delay=DT,
           sic_delay_steps=SIC_DELAY_STEPS, seed=0)


def _ms(x):
    """Strip units to a flat float64 ndarray (a recorded-trace mantissa)."""
    return np.asarray(u.get_mantissa(x), dtype=float)


def _mean_trace(x):
    """Mean over the neuron axis of a (n_steps, n) brainpy trace -> (n_steps,)."""
    a = _ms(x)
    return a.mean(axis=1) if a.ndim == 2 else a


def _nest_mean_trace(events, var):
    """Per-time mean of a NEST multimeter variable over its recorded senders."""
    times = np.asarray(events['times'], dtype=float)
    vals = np.asarray(events[var], dtype=float)
    uniq = np.unique(times)
    return np.array([vals[times == t].mean() for t in uniq])


def _nest_small_network(cfg):
    """Build the same deterministic tripartite net in live NEST; return trace means."""
    nest.set_verbosity('M_ERROR')
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': cfg['dt']})
    npar = {'tau_syn_ex': cfg['tau_syn_ex'], 'I_e': cfg['I_e']}
    pre = nest.Create('aeif_cond_alpha_astro', cfg['n_neurons'], params=npar)
    post = nest.Create('aeif_cond_alpha_astro', cfg['n_neurons'], params=npar)
    astro = nest.Create('astrocyte_lr_1994', cfg['n_astro'],
                        params={'delta_IP3': cfg['delta_IP3'], 'IP3': cfg['IP3_0']})
    nest.TripartiteConnect(
        pre, post, astro,
        conn_spec={'rule': 'pairwise_bernoulli', 'p': cfg['p_primary']},
        third_factor_conn_spec={'rule': 'third_factor_bernoulli_with_pool',
                                'p': cfg['p_third'], 'pool_size': cfg['pool_size'],
                                'pool_type': cfg['pool_type']},
        syn_specs={'primary': {'synapse_model': 'static_synapse',
                               'weight': cfg['w_primary'], 'delay': cfg['conn_delay']},
                   'third_in': {'synapse_model': 'static_synapse',
                                'weight': cfg['w_third_in'], 'delay': cfg['conn_delay']},
                   'third_out': {'synapse_model': 'sic_connection', 'weight': cfg['w_a2n']}})
    mm_pre = nest.Create('multimeter', params={'record_from': ['V_m'], 'interval': cfg['dt']})
    mm_post = nest.Create('multimeter', params={'record_from': ['V_m', 'I_SIC'], 'interval': cfg['dt']})
    mm_a = nest.Create('multimeter', params={'record_from': ['IP3', 'Ca_astro'], 'interval': cfg['dt']})
    nest.Connect(mm_pre, pre)
    nest.Connect(mm_post, post)
    nest.Connect(mm_a, astro)
    nest.Simulate(cfg['sim_time'])
    return {'V_pre': _nest_mean_trace(mm_pre.events, 'V_m'),
            'V_post': _nest_mean_trace(mm_post.events, 'V_m'),
            'I_SIC': _nest_mean_trace(mm_post.events, 'I_SIC'),
            'IP3': _nest_mean_trace(mm_a.events, 'IP3'),
            'Ca': _nest_mean_trace(mm_a.events, 'Ca_astro')}


class TestAstrocyteSmallNetworkLaw(unittest.TestCase):
    """Structural / dynamical invariants that need no NEST (always run)."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_loop_lowers_with_finite_traces_and_live_sic(self):
        """The net lowers under for_loop; traces are finite; IP3 climbs; SIC fires."""
        out = run(**DET)
        n = int(round(DET['sim_time'] / DT))
        ip3 = _ms(out['IP3']); isic = _ms(out['I_SIC'])
        self.assertEqual(ip3.shape, (n, DET['n_astro']))
        self.assertEqual(isic.shape, (n, DET['n_neurons']))
        for key in ('V_pre', 'V_post', 'IP3', 'Ca', 'I_SIC'):
            self.assertTrue(np.all(np.isfinite(_ms(out[key]))), f'{key} must be finite')
        self.assertGreater(float(ip3.max()), DET['IP3_0'], 'IP3 climbs with presynaptic drive')
        self.assertGreater(float(isic.max()), 0.0, 'the SIC arm delivers current to post')

    def test_block_symmetry_identical_neurons(self):
        """Block symmetry: all pre / all post / all astro share an identical trace."""
        out = run(**DET)
        for key in ('V_pre', 'IP3', 'I_SIC'):
            a = _ms(out[key])
            self.assertTrue(np.allclose(a, a[:, :1], atol=1e-9),
                            f'{key} columns must be identical under block symmetry')


@requires_nest
class TestAstrocyteSmallNetworkParity(unittest.TestCase):
    """Mean V / IP3 / Ca / I_SIC track live NEST per-sample (deterministic net)."""

    def setUp(self):
        brainstate.environ.set(dt=DT * u.ms)

    def tearDown(self):
        jax.clear_caches()
        gc.collect()

    def test_loop_traces_match_nest(self):
        """Driver V_pre (CAT_A), astro IP3/Ca (ASTRO_TOL) and I_SIC (SIC_TOL) match NEST.

        ``V_post`` is excluded from sample-wise parity (chaotic spiking output; see
        the module docstring); the loop-coupling signals are the faithful check.
        """
        nref = _nest_small_network(DET)
        out = run(**DET)
        bp = {k: _mean_trace(out[k]) for k in ('V_pre', 'IP3', 'Ca', 'I_SIC')}
        self.assertGreater(float(np.max(nref['I_SIC'])), 0.0)   # SIC arm fired in NEST
        for nm, tol in (('V_pre', CAT_A), ('IP3', ASTRO_TOL), ('Ca', ASTRO_TOL),
                        ('I_SIC', SIC_TOL)):
            ref, cand = nref[nm], bp[nm]
            n = min(ref.size, cand.size)
            compare_trace(ref[:n], cand[:n], tol=tol, metric=nm).assert_()


if __name__ == '__main__':
    unittest.main()
