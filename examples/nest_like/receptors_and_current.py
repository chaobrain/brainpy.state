# examples/nest_like/receptors_and_current.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Three-compartment model with different receptor types + a current injection.

Ports NEST's ``pynest/examples/compartmental_model/receptors_and_current.py`` to
the Simulator API. A single passive ``cm_default`` tree — a soma with two
dendrites — carries a **different receptor on each compartment**, and a steady
current is injected into one dendrite:

* compartment 0 (soma)  — ``GABA``       receptor (index 0, inhibitory);
* compartment 1 (dend1) — ``AMPA``       receptor (index 1, fast excitatory);
* compartment 2 (dend2) — ``AMPA_NMDA``  receptor (index 2, exc + NMDA).

Three ``spike_generator``s drive the three receptors (by receptor index), and a
``dc_generator`` injects 1 pA into compartment 1. **Spike receptors are addressed
by receptor index; a current generator's ``receptor_type`` is the compartment
index** (NEST's convention — "the receptor type is the compartment index" for
current inputs). The recorded per-compartment voltages show each receptor's
distinct signature: the AMPA EPSPs on dend1 (riding on the dc offset), the slower
NMDA-flavoured EPSPs on dend2, and the GABA IPSPs hyperpolarising the soma — all
attenuated as they spread electrotonically to the other compartments.

The live-NEST parity test
(``brainpy_state/_nest/_validation/receptors_and_current_test.py``) drives a real
``cm_default`` with this identical wiring and confirms the three compartment
voltages match.

Run:  PYTHONPATH=. python examples/nest_like/receptors_and_current.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy_state import Simulator, cm_default, spike_generator, dc_generator, multimeter

DT = 0.1        # resolution [ms]
T_SIM = 400.0   # total simulation time [ms]

# Compartment parameters, verbatim from NEST receptors_and_current.py. The whole
# tree is passive (no Na/K channels): a soma with two identical dendrites.
SOMA_PARAMS = {'C_m': 10.0, 'g_C': 0.0, 'g_L': 1.0, 'e_L': -70.0, 'v_comp': -70.0}
DEND_PARAMS = {'C_m': 0.1, 'g_C': 0.1, 'g_L': 0.1, 'e_L': -70.0, 'v_comp': -70.0}
V_TH = -50.0  # spike-detection threshold [mV]

# Compartments: soma (0) with two child dendrites (1, 2).
COMPARTMENTS = [
    {'parent_idx': -1, 'params': SOMA_PARAMS},
    {'parent_idx': 0, 'params': DEND_PARAMS},
    {'parent_idx': 0, 'params': DEND_PARAMS},
]
# One receptor per compartment; add-order fixes the receptor index (NEST syn_idx).
RECEPTORS = [
    {'comp_idx': 0, 'receptor_type': 'GABA'},                                          # idx 0 (soma)
    {'comp_idx': 1, 'receptor_type': 'AMPA',
     'params': {'tau_r_AMPA': 0.2, 'tau_d_AMPA': 3.0, 'e_AMPA': 0.0}},                 # idx 1 (dend1)
    {'comp_idx': 2, 'receptor_type': 'AMPA_NMDA'},                                     # idx 2 (dend2)
]
SYN_GABA, SYN_AMPA, SYN_NMDA = 0, 1, 2   # receptor indices (== NEST syn_idx)

# Per-receptor spike trains (receptor index, [spike times ms], weight nS).
SPIKE_TRAINS = [
    (SYN_AMPA, [101.0, 105.0, 106.0, 110.0, 150.0], 0.1),                              # AMPA -> dend1
    (SYN_NMDA, [115.0, 155.0, 160.0, 162.0, 170.0, 254.0, 260.0, 272.0, 278.0], 0.2),  # AMPA_NMDA -> dend2
    (SYN_GABA, [250.0, 255.0, 260.0, 262.0, 270.0], 0.3),                              # GABA -> soma
]
DELAY = 0.5         # spike connection delay [ms]
DC_COMPARTMENT = 1  # current injected into compartment 1 (NEST receptor_type = comp index)
DC_AMPLITUDE = 1.0  # injected current [pA]

RECORDABLES = ['v_comp0', 'v_comp1', 'v_comp2']


def build(simtime=T_SIM):
    """Build the three-compartment, three-receptor model with the dc injection wired.

    Parameters
    ----------
    simtime : float, optional
        Simulation horizon in ms. Default :data:`T_SIM`.

    Returns
    -------
    sim : Simulator
    mm : NodeView
        Multimeter recording the three compartment voltages ``v_comp0/1/2``
        (read via ``res.trace(mm, name)``).
    simtime : float
    """
    sim = Simulator(dt=DT * u.ms)
    cm = sim.create(cm_default, 1,
                    params={'compartments': COMPARTMENTS, 'receptors': RECEPTORS, 'V_th': V_TH})

    # Three spike trains, each routed to its receptor by receptor index.
    for rtype, times, w in SPIKE_TRAINS:
        sg = sim.create(spike_generator, spike_times=np.asarray(times) * u.ms)
        sim.connect(sg, cm, receptor_type=rtype, weight=w * u.nS, delay=DELAY * u.ms)

    # Steady current into compartment 1 (a current generator's receptor_type is the
    # compartment index). The injection rides cm_default's one-step current buffer,
    # matching NEST's 0.1 ms dc connection delay.
    dcg = sim.create(dc_generator, amplitude=DC_AMPLITUDE * u.pA)
    sim.connect(dcg, cm, receptor_type=DC_COMPARTMENT)

    mm = sim.create(multimeter, record_from=RECORDABLES, interval=DT * u.ms)
    sim.connect(mm, cm)
    return sim, mm, simtime


def run_traces(simtime=T_SIM):
    """Run the demo and return ``(t_ms, traces)``.

    ``traces`` maps each name in :data:`RECORDABLES` to its 1-D voltage trace (mV).
    """
    sim, mm, _t = build(simtime)
    res = sim.simulate(simtime * u.ms)
    t = np.asarray(u.get_mantissa(res.times / u.ms))
    traces = {name: np.asarray(u.get_mantissa(res.trace(mm, name) / u.mV)).reshape(-1)
              for name in RECORDABLES}
    return t, traces


def main():
    print("Three-compartment cm_default: GABA/AMPA/AMPA_NMDA receptors + dc (brainpy.state)")
    t, traces = run_traces()
    for name, label in (('v_comp0', 'soma (GABA)'), ('v_comp1', 'dend1 (AMPA+dc)'),
                        ('v_comp2', 'dend2 (AMPA_NMDA)')):
        v = traces[name]
        print(f"  {label:20s}: V [{v.min():7.3f}, {v.max():7.3f}] mV")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 4.5))
        for name, col in (('v_comp0', 'b'), ('v_comp1', 'r'), ('v_comp2', 'g')):
            plt.plot(t, traces[name], c=col, label=name)
        plt.xlabel(r"$t$ (ms)")
        plt.ylabel("membrane potential [mV]")
        plt.title("cm_default: different receptor types + current injection")
        plt.legend(loc=0)
        plt.tight_layout()
        plt.savefig("examples/nest_like/receptors_and_current.png", dpi=100)
        print("  wrote examples/nest_like/receptors_and_current.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
