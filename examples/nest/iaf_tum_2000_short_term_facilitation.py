# examples/nest/iaf_tum_2000_short_term_facilitation.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Short-term facilitation in iaf_tum_2000 — NEST-style port.

Port of NEST's ``iaf_tum_2000_short_term_facilitation.py``, reproducing Figure 1B
of Tsodyks, Pawelzik & Markram (1998). Like the depression example, the
``iaf_tum_2000`` neuron integrates Tsodyks-Markram short-term plasticity in the
presynaptic neuron, but here a *small* ``U`` together with a non-zero
facilitation time constant ``tau_fac`` makes the release probability ``u`` grow
across successive spikes faster than the resources ``x`` deplete. The synaptic
efficacy ``u * x`` therefore *increases* over the train — short-term
facilitation.

Two ``iaf_tum_2000`` neurons are created. The presynaptic neuron is driven by a
constant current (a ``dc_generator``) so it fires a regular ~20 Hz train; a plain
``static_synapse`` on ``receptor_type=1`` carries the graded released efficacy
``weight * (u * x)`` to the postsynaptic neuron, whose sub-threshold membrane
potential is recorded. The successive EPSP amplitudes *grow* then saturate — the
signature of short-term facilitation.

The presynaptic STP efficacy is delivered through the Simulator's
presynaptic-emission seam (``connect(pre, post, receptor_type=1)`` reads the
released efficacy rather than the binary spike); see also
``iaf_tum_2000_short_term_depression.py``.

Run:  PYTHONPATH=. python examples/nest/iaf_tum_2000_short_term_facilitation.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import braintools
import brainunit as u

from brainpy_state import Simulator, iaf_tum_2000, dc_generator, voltmeter

# Membrane parameters (Tsodyks-Pawelzik-Markram 1998, Fig 1B; facilitation regime).
TAU_M = 60.0                       # membrane time constant [ms]
R_M = 1.0                          # membrane input resistance [GΩ]
C_M = TAU_M / R_M                  # membrane capacitance [pF] (= 60)
V_TH = 15.0                        # threshold potential [mV]
V_RESET = 0.0                      # reset / resting potential [mV]
T_REF = 2.0                        # refractory period [ms]

# Short-term plasticity parameters (facilitation: small U, non-zero tau_fac).
X0 = 1.0                           # initial readily-releasable fraction
U0 = 0.0                           # initial release probability
U = 0.03                           # increase of u per spike
TAU_PSC = 1.5                      # PSC decay constant [ms]
TAU_REC = 130.0                    # recovery from depression [ms]
TAU_FAC = 530.0                    # facilitation time constant [ms]

# Connection and stimulation.
WEIGHT = 1540.0                    # synaptic weight [pA]
DELAY = 0.1                        # synaptic delay [ms]
STIM_START = 50.0                  # DC start [ms]
STIM_END = 1050.0                  # DC stop [ms]
T_SIM = 1200.0                     # simulation time [ms]
DT = 0.1                           # resolution [ms]
_F = 20.0 / 1000.0                 # target firing frequency [1/ms]
# DC amplitude tuned so the presynaptic neuron fires at ~20 Hz (NEST formula).
DC_AMP = V_TH * C_M / TAU_M / (1.0 - np.exp(-(1.0 / _F - T_REF) / TAU_M))


def _neuron_params():
    return dict(
        C_m=C_M * u.pF, tau_m=TAU_M * u.ms,
        tau_syn_ex=TAU_PSC * u.ms, tau_syn_in=TAU_PSC * u.ms,
        V_th=V_TH * u.mV, V_reset=V_RESET * u.mV, E_L=V_RESET * u.mV,
        t_ref=T_REF * u.ms, U=U, tau_psc=TAU_PSC * u.ms,
        tau_rec=TAU_REC * u.ms, tau_fac=TAU_FAC * u.ms, x=X0, u=U0,
        V_initializer=braintools.init.Constant(V_RESET * u.mV),
    )


def build(simtime=T_SIM):
    """Build the two-neuron short-term-facilitation network.

    Returns
    -------
    sim : Simulator
    vm : NodeView
        Voltmeter handle observing the postsynaptic neuron (``res.trace(vm, 'V_m')``).
    post : NodeView
        The postsynaptic neuron.
    simtime : float
        Simulation horizon in ms.
    """
    sim = Simulator(dt=DT * u.ms)
    params = _neuron_params()
    pre = sim.create(iaf_tum_2000, 1, params=params)
    post = sim.create(iaf_tum_2000, 1, params=params)
    dc = sim.create(dc_generator, amplitude=DC_AMP * u.pA,
                    start=STIM_START * u.ms, stop=STIM_END * u.ms)
    vm = sim.create(voltmeter)
    sim.connect(dc, pre)                                  # drive the presynaptic neuron
    sim.connect(pre, post, receptor_type=1,               # TSODYKS: deliver weight*(u*x)
                weight=WEIGHT * u.pA, delay=DELAY * u.ms)
    sim.connect(vm, post)                                 # observe the postsynaptic V_m
    return sim, vm, post, simtime


def run_traces(simtime=T_SIM):
    """Return ``(t_ms, v_post_mV)`` for the postsynaptic membrane potential."""
    sim, vm, _post, _t = build(simtime)
    res = sim.simulate(simtime * u.ms)
    t = np.asarray(u.get_mantissa(res.times / u.ms))
    v = np.asarray(u.get_mantissa(res.trace(vm, 'V_m') / u.mV)).reshape(-1)
    return t, v


def main():
    print('iaf_tum_2000 short-term facilitation (brainpy.state, Fig 1B)')
    print(f'  DC amplitude {DC_AMP:.2f} pA -> ~20 Hz presynaptic train')
    t, v = run_traces()
    print(f'  post V_m: rest {v[0]:.3f} mV, peak {v.max():.3f} mV')

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(t, v, color='k', lw=0.8)
        plt.xlabel('time (ms)'); plt.ylabel('post V_m (mV)')
        plt.title('iaf_tum_2000 — short-term facilitation')
        plt.tight_layout()
        plt.savefig('examples/nest/iaf_tum_2000_short_term_facilitation.png', dpi=100)
        print('  wrote examples/nest/iaf_tum_2000_short_term_facilitation.png')
    except ImportError:
        print('  (matplotlib not installed; skipping plot)')


if __name__ == '__main__':
    main()
