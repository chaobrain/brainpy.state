# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Gap junctions: inhibitory network synchronization (ported NEST example).

A port of NEST's ``gap_junctions_inhibitory_network.py`` (Hahne et al. 2015, test case 2).
A recurrent inhibitory network of ``hh_psc_alpha_gap`` cells -- random ``fixed_indegree``
static inhibition, an all-to-all excitatory Poisson drive, and random initial
``V_m ~ U[-80, -40] mV`` -- is coupled by a symmetric random gap-junction graph. Without
gap junctions the balanced network is asynchronous-irregular; as the gap weight rises the
network synchronizes (``gap_weight`` 0.0 async, ~0.54 bistable, 0.7 synchronous).

The gap junction is realized as the explicit one-step-lagged difference current
``I_gap,i = sum_j g_ij (V_j[n-1] - V_i[n-1])`` (NEST's ``use_wfr=False`` regime; no
waveform relaxation). The distributional synchronization parity against live NEST lives in
``gap_junction_inhibitory_network_parity_test.py`` (``@requires_nest``): the Golomb-Rinzel
population coherence matches NEST to within a few percent at both async and synchronous
gap weights.

Run ``python examples/nest_like/gap_junctions_inhibitory_network.py`` to simulate and plot a
spike raster at a few gap weights.
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u

from brainpy.state import (Simulator, hh_psc_alpha_gap, voltmeter, spike_recorder,
                           poisson_generator, all_to_all, fixed_indegree)
from brainpy_state._nest_synapse.gap_junction import gap_junction
from examples.nest_like.gap_junctions_two_neurons import resting_gating

INH = 50             # inhibitory in-degree per neuron (demo scale; reduced for tiny runs)
GAPK = 30            # gap in-degree pre-symmetrization (~2x edges after make_symmetric)
DT = 0.05            # ms
J_EXC = 300.0        # pA, Poisson drive weight
J_INH = -50.0        # pA, recurrent inhibition weight
RATE = 500.0         # Hz, Poisson rate
DELAY = 1.0          # ms, chemical-synapse delay


def run_network(gap_weight, seed=0, *, n_neuron=500, inh=INH, gap_k=GAPK,
                T=501.0, dt=DT, record_spikes=True):
    """Simulate the inhibitory gap network.

    Returns ``{'V': (S, N) mV, 't': (S,), 'spikes': (S, N) or None}``. ``V`` feeds the
    synchrony measure; ``spikes`` (binary, per recorded neuron) drives the raster plot.
    """
    sim = Simulator(dt=dt * u.ms)
    v_init = jax.random.uniform(jax.random.PRNGKey(seed), (n_neuron,),
                                minval=-80.0, maxval=-40.0)
    nrn = sim.create(hh_psc_alpha_gap, n_neuron, params={
        'V_m_init': v_init * u.mV, 'I_e': 0.0 * u.pA, **resting_gating()})
    pg = sim.create(poisson_generator, rate=RATE * u.Hz)
    sim.connect(nrn, nrn, weight=J_INH * u.pA, delay=DELAY * u.ms,
                rule=fixed_indegree(min(inh, n_neuron - 1)), allow_multapses=True,
                seed=seed + 7)
    sim.connect(pg, nrn, weight=J_EXC * u.pA, delay=DELAY * u.ms, rule=all_to_all)
    if gap_weight > 0:
        sim.connect(nrn, nrn, weight=gap_weight * u.nS, synapse=gap_junction, comm='dense',
                    rule=fixed_indegree(min(gap_k, n_neuron - 1)), allow_autapses=False,
                    seed=seed + 99)
    vm = sim.create(voltmeter)
    sim.connect(vm, nrn)
    sr = sim.create(spike_recorder) if record_spikes else None
    if sr is not None:
        sim.connect(nrn, sr)
    res = sim.simulate(T * u.ms)
    V = np.asarray(u.get_mantissa(res.trace(vm, 'V_m') / u.mV))
    out = {'V': V, 't': np.arange(V.shape[0]) * dt, 'spikes': None}
    if sr is not None:
        out['spikes'] = np.asarray(res.spikes(sr))
    return out


def golomb_chi(V, *, skip_ms=150.0, dt=DT):
    r"""Golomb-Rinzel population coherence ``sqrt(Var_t<V> / mean_i Var_t V_i)`` (0 async)."""
    V = V[int(skip_ms / dt):]
    den = float(V.var(axis=0).mean())
    return float(np.sqrt(V.mean(axis=1).var() / den)) if den > 0 else 0.0


def main():  # pragma: no cover - manual full-scale demo driver (I/O + matplotlib)
    weights = [0.0, 0.3, 0.7]
    results = {}
    for gw in weights:
        out = run_network(gw, seed=1, n_neuron=500, T=501.0)
        chi = golomb_chi(out['V'])
        n_spk = int(np.sum(out['spikes'] > 0))
        rate = 1000.0 * n_spk / 501.0 / 500
        results[gw] = out
        print(f'gap_weight={gw:.2f}: chi={chi:.3f}  mean rate={rate:.1f} spk/s')
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('  (matplotlib not installed; skipping raster plot)')
        return
    fig, axes = plt.subplots(len(weights), 1, figsize=(8, 8), sharex=True)
    for ax, gw in zip(axes, weights):
        spk = results[gw]['spikes']; t = results[gw]['t']
        rows, cols = np.nonzero(spk > 0)
        ax.plot(t[rows], cols, '|', ms=2, color='k')
        ax.set_ylabel(f'g={gw}\nneuron')
    axes[-1].set_xlabel('time (ms)')
    axes[0].set_title('Gap junctions: inhibitory network (async -> synchronous)')
    plt.tight_layout()
    plt.savefig('examples/nest_like/gap_junctions_inhibitory_network.png', dpi=100)
    print('  saved examples/nest_like/gap_junctions_inhibitory_network.png')


if __name__ == '__main__':  # pragma: no cover
    main()
