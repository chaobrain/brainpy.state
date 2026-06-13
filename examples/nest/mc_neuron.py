# examples/nest/mc_neuron.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Three-compartment neuron (``iaf_cond_alpha_mc``) — NEST-style port.

Ports NEST's ``pynest/examples/mc_neuron.py`` to the Simulator API. The
``iaf_cond_alpha_mc`` model has a soma, a proximal and a distal compartment, each
with its own membrane potential and its own excitatory / inhibitory alpha-shaped
conductances. The model exposes **nine receptors**: six spike receptors (one
exc + one inh per compartment) and three per-compartment current receptors. A
connection's ``receptor_type`` uniquely selects the target compartment *and*
channel, exactly as in NEST.

Three stimulation paradigms are shown, distal → proximal → soma in time:

1. **Per-compartment current** — a ``dc_generator`` drives each compartment in
   turn (distal +100 pA at 50–100 ms, proximal −50 pA at 150–200 ms, soma
   +50 pA at 250–300 ms), routed through current receptors 7/8/9.
2. **Per-compartment spikes** — an excitatory and an inhibitory ``spike_generator``
   hit each compartment (distal ~400 ms, proximal ~500 ms, soma ~600 ms), routed
   through spike receptors 1–6, so you can watch ``g_ex`` / ``g_in`` rise in only
   the targeted compartment.
3. **Somatic rheobase** — from 700 ms a steady 150 pA is injected into the soma,
   driving the neuron over threshold so it emits output spikes.

NEST's published example produces paradigm 3 by setting ``n.soma = {'I_e': 150.0}``
midway through the run. The Simulator lowers the whole simulation into one compiled
loop, so a parameter cannot be changed mid-run; the faithful Simulator-API
equivalent is a ``step_current_generator`` that steps the soma current to 150 pA at
700 ms (a timed device current into the soma current receptor). The live-NEST
parity test (``brainpy_state/_nest/_validation/mc_neuron_test.py``) drives a real
``iaf_cond_alpha_mc`` with this identical wiring and confirms the per-compartment
``V_m`` and ``g_ex`` / ``g_in`` traces match.

Run:  PYTHONPATH=. python examples/nest/mc_neuron.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import saiunit as u

from brainpy_state import (Simulator, iaf_cond_alpha_mc, dc_generator,
                           step_current_generator, spike_generator,
                           multimeter, spike_recorder)

DT = 0.1            # resolution [ms]
T_SIM = 1000.0      # total simulation time [ms]
RHEO_START = 700.0  # soma rheobase onset [ms]
RHEO_AMP = 150.0    # soma rheobase amplitude [pA]
SPIKE_WEIGHT = 1.0  # conductance increment per incoming spike [nS] (NEST default)

# Non-default neuron parameters (verbatim from NEST mc_neuron.py).
NEURON_PARAMS = dict(
    V_th=-60.0 * u.mV,
    V_reset=-65.0 * u.mV,
    t_ref=10.0 * u.ms,
    g_sp=5.0 * u.nS,                                  # somato-proximal coupling
    soma={'g_L': 12.0 * u.nS},                        # somatic leak
    proximal={'tau_syn_ex': 1.0 * u.ms, 'tau_syn_in': 5.0 * u.ms},
    distal={'C_m': 90.0 * u.pF},                      # distal capacitance
)

# Per-compartment current pulses: (receptor_type, amplitude pA, start ms, stop ms).
# Receptors 7/8/9 == soma_curr / proximal_curr / distal_curr.
CURRENT_PULSES = [
    (7, 50.0, 250.0, 300.0),    # soma     +50 pA
    (8, -50.0, 150.0, 200.0),   # proximal -50 pA
    (9, 100.0, 50.0, 100.0),    # distal  +100 pA
]

# Per-compartment spike trains: (receptor_type, [spike times ms]).
# Receptors 1..6 == soma_exc, soma_inh, proximal_exc, proximal_inh,
# distal_exc, distal_inh.
SPIKE_TRAINS = [
    (1, [600.0, 620.0]),   # soma excitatory
    (2, [610.0, 630.0]),   # soma inhibitory
    (3, [500.0, 520.0]),   # proximal excitatory
    (4, [510.0, 530.0]),   # proximal inhibitory
    (5, [400.0, 420.0]),   # distal excitatory
    (6, [410.0, 430.0]),   # distal inhibitory
]

# The nine analog recordables NEST's example plots (compartment × quantity).
RECORDABLES = ['V_m.s', 'V_m.p', 'V_m.d',
               'g_ex.s', 'g_ex.p', 'g_ex.d',
               'g_in.s', 'g_in.p', 'g_in.d']


def build(simtime=T_SIM):
    """Build the three-compartment-neuron Simulator with all three paradigms wired.

    Parameters
    ----------
    simtime : float, optional
        Simulation horizon in ms. Default :data:`T_SIM`.

    Returns
    -------
    sim : Simulator
    mm : NodeView
        Multimeter recording the nine per-compartment ``V_m`` / ``g_ex`` / ``g_in``
        quantities (read via ``res.trace(mm, name)``).
    sr : NodeView
        Spike recorder on the soma output (``res.n_events(sr)``).
    simtime : float
    """
    sim = Simulator(dt=DT * u.ms)
    neuron = sim.create(iaf_cond_alpha_mc, 1, params=NEURON_PARAMS)

    # Paradigm 1: one dc_generator per compartment, routed by current receptor.
    for rtype, amp, start, stop in CURRENT_PULSES:
        dc = sim.create(dc_generator, amplitude=amp * u.pA,
                        start=start * u.ms, stop=stop * u.ms)
        sim.connect(dc, neuron, receptor_type=rtype)

    # Paradigm 2: one exc + one inh spike train per compartment, routed by spike
    # receptor. The conductance weight (nS) lands in exactly one channel.
    for rtype, times in SPIKE_TRAINS:
        sg = sim.create(spike_generator, spike_times=np.asarray(times) * u.ms)
        sim.connect(sg, neuron, receptor_type=rtype, weight=SPIKE_WEIGHT * u.nS)

    # Paradigm 3: somatic rheobase — step the soma current to 150 pA at 700 ms.
    rheo = sim.create(step_current_generator,
                      amplitude_times=[RHEO_START] * u.ms,
                      amplitude_values=[RHEO_AMP] * u.pA)
    sim.connect(rheo, neuron, receptor_type=7)   # soma_curr

    mm = sim.create(multimeter, record_from=RECORDABLES, interval=DT * u.ms)
    sr = sim.create(spike_recorder)
    sim.connect(mm, neuron)
    sim.connect(neuron, sr)
    return sim, mm, sr, simtime


def run_traces(simtime=T_SIM):
    """Run the demo and return ``(t_ms, traces, n_spikes)``.

    ``traces`` is a dict mapping each recordable name in :data:`RECORDABLES` to its
    1-D trace (``V_m.*`` in mV, ``g_*`` in nS).
    """
    sim, mm, sr, _t = build(simtime)
    res = sim.simulate(simtime * u.ms)
    t = np.asarray(u.get_mantissa(res.times / u.ms))
    traces = {}
    for name in RECORDABLES:
        unit = u.mV if name.startswith('V_m') else u.nS
        traces[name] = np.asarray(u.get_mantissa(res.trace(mm, name) / unit)).reshape(-1)
    return t, traces, int(res.n_events(sr))


def main():
    print("Three-compartment iaf_cond_alpha_mc neuron (brainpy.state)")
    t, traces, n = run_traces()
    print(f"  output spikes (soma rheobase): {n}")
    for c, label in (('s', 'soma'), ('p', 'proximal'), ('d', 'distal')):
        v = traces[f'V_m.{c}']
        print(f"  {label:9s}: V_m [{v.min():6.2f}, {v.max():6.2f}] mV, "
              f"g_ex max {traces[f'g_ex.{c}'].max():.3f} nS, "
              f"g_in max {traces[f'g_in.{c}'].max():.3f} nS")

    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for c, name in (('s', 'soma'), ('p', 'proximal'), ('d', 'distal')):
            ax1.plot(t, traces[f'V_m.{c}'], label=name)
        ax1.set_ylabel("membrane potential [mV]")
        ax1.set_title("Responses of iaf_cond_alpha_mc neuron")
        ax1.legend(loc="lower right", fontsize=8)
        for c, col in (('s', 'C0'), ('p', 'C1'), ('d', 'C2')):
            ax2.plot(t, traces[f'g_ex.{c}'], color=col, ls='-', label=f'g_ex.{c}')
            ax2.plot(t, traces[f'g_in.{c}'], color=col, ls='--', label=f'g_in.{c}')
        ax2.set_xlabel("time [ms]")
        ax2.set_ylabel("synaptic conductance [nS]")
        ax2.legend(loc="upper right", fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig("examples/nest/mc_neuron.png", dpi=100)
        print("  wrote examples/nest/mc_neuron.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
