# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Gap junctions: two-neuron synchronization (ported NEST example).

A port of NEST's ``gap_junctions_two_neurons.py``. Two ``hh_psc_alpha_gap`` cells are
driven by a constant ``I_e = 100 pA``; one is perturbed to ``V_m = -10 mV`` while the
other rests. A single symmetric ``gap_junction`` connection (``g = 0.5 nS``) couples them
electrically, and over a few hundred milliseconds the two membrane potentials synchronize.

The port realizes the gap junction as an explicit one-step-lagged **difference current**
``I_gap,i = sum_j g_ij (V_j[n-1] - V_i[n-1])`` deposited into the post cell's current
channel -- NEST's ``use_wfr=False`` regime, with no waveform relaxation. The 2-neuron
live-NEST micro-parity (``gap_junction_parity_test.py``) confirms this reproduces NEST to
machine precision between spikes (the only difference is an O(dt) AP-edge timing jitter).

**Initial conditions.** NEST sets the gating variables once at construction (equilibrium
at the resting default) and a later ``V_m`` change does *not* recompute them, so the
perturbed cell carries resting gating. The port's convention is ``eq(V_m_init)`` per
neuron, so :func:`resting_gating` overrides the gating to the resting equilibrium to
reproduce NEST's exact ICs.

Run ``python examples/nest_like/gap_junctions_two_neurons.py`` to simulate and plot.
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import jax.numpy as jnp
import numpy as np
import brainunit as u

from brainpy.state import Simulator, hh_psc_alpha_gap, voltmeter, all_to_all
from brainpy.state import gap_junction

I_E = 100.0          # pA, constant drive
V_PERTURB = -10.0    # mV, the perturbed cell's initial voltage
GAP_WEIGHT = 0.5     # nS, gap conductance
DT = 0.05            # ms
T = 351.0            # ms
VR = -69.60401191631222  # mV, hh_psc_alpha_gap resting default (NEST V_m init)


def equilibrium_gating(V):
    r"""Equilibrium gating ``(m, h, n, p)`` of ``hh_psc_alpha_gap`` at voltage ``V`` (mV).

    Steady states :math:`x_\infty = \alpha_x / (\alpha_x + \beta_x)` for the
    Mancilla et al. (2007) rate functions used by ``hh_psc_alpha_gap``. Used only
    to set matched initial conditions; not part of the simulation loop.
    """
    alpha_m = 40.0 * (V - 75.5) / (1.0 - np.exp(-(V - 75.5) / 13.5))
    beta_m = 1.2262 / np.exp(V / 42.248)
    alpha_h = 0.0035 / np.exp(V / 24.186)
    beta_h = 0.017 * (51.25 + V) / (1.0 - np.exp(-(51.25 + V) / 5.2))
    alpha_n = 0.014 * (V + 44.0) / (1.0 - np.exp(-(V + 44.0) / 2.3))
    beta_n = 0.0043 / np.exp((V + 44.0) / 34.0)
    alpha_p = (V - 95.0) / (1.0 - np.exp(-(V - 95.0) / 11.8))
    beta_p = 0.025 / np.exp(V / 22.222)
    return (alpha_m / (alpha_m + beta_m), alpha_h / (alpha_h + beta_h),
            alpha_n / (alpha_n + beta_n), alpha_p / (alpha_p + beta_p))


def resting_gating():
    """``hh_psc_alpha_gap`` gating at the resting default (NEST's frozen-gating ICs).

    Returns a params dict so a perturbed ``V_m`` does not drag the gating with it -- the
    cell starts with resting gating exactly as in NEST's ``Create`` + ``SetStatus(V_m)``.
    """
    m, h, n, p = equilibrium_gating(VR)
    return dict(Act_m_init=m, Inact_h_init=h, Act_n_init=n, Inact_p_init=p)


def run_two_neuron(gap_weight=GAP_WEIGHT, *, v_perturb=V_PERTURB, T=T, dt=DT):
    """Simulate the coupled pair; return ``{'t': (S,), 'V': (S, 2)}`` membrane traces (mV)."""
    sim = Simulator(dt=dt * u.ms)
    pop = sim.create(hh_psc_alpha_gap, 2, params={
        'V_m_init': jnp.array([v_perturb, VR]) * u.mV, 'I_e': I_E * u.pA,
        **resting_gating()})
    vm = sim.create(voltmeter)
    if gap_weight > 0:
        sim.connect(pop, pop, rule=all_to_all, weight=gap_weight * u.nS,
                    synapse=gap_junction, comm='dense', allow_autapses=False)
    sim.connect(vm, pop)
    res = sim.simulate(T * u.ms)
    V = np.asarray(u.get_mantissa(res.trace(vm, 'V_m') / u.mV))
    t = np.arange(V.shape[0]) * dt
    return {'t': t, 'V': V}


def synchrony_gap(V, *, window_ms=20.0, dt=DT):
    """RMS membrane gap ``|V_0 - V_1|`` over the first / last ``window_ms`` (convergence)."""
    w = int(window_ms / dt)
    d = np.abs(V[:, 0] - V[:, 1])
    early = float(np.sqrt(np.mean(d[:w] ** 2)))
    late = float(np.sqrt(np.mean(d[-w:] ** 2)))
    return early, late


def main():  # pragma: no cover - manual demo driver (I/O + matplotlib)
    out = run_two_neuron()
    early, late = synchrony_gap(out['V'])
    print(f'membrane gap RMS: start {early:.2f} mV -> end {late:.3f} mV (synchronized)')
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('  (matplotlib not installed; skipping plot)')
        return
    t, V = out['t'], out['V']
    plt.figure(figsize=(8, 4))
    plt.plot(t, V[:, 0], label='neuron 0 (perturbed to -10 mV)', color='C0', lw=0.8)
    plt.plot(t, V[:, 1], label='neuron 1 (rest)', color='C1', lw=0.8)
    plt.xlabel('time (ms)'); plt.ylabel('V_m (mV)')
    plt.title('Gap junctions: two-neuron synchronization (g = 0.5 nS)')
    plt.legend(); plt.tight_layout()
    plt.savefig('examples/nest_like/gap_junctions_two_neurons.png', dpi=100)
    print('  saved examples/nest_like/gap_junctions_two_neurons.png')


if __name__ == '__main__':  # pragma: no cover
    main()
