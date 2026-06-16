# examples/nest_like/two_comps.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Two-compartment models with active vs. passive dendrites (``cm_default``).

Ports NEST's ``pynest/examples/compartmental_model/two_comps.py`` to the
Simulator API. ``cm_default`` is the general user-defined-morphology
multi-compartment neuron: you build a tree of compartments (each with its own
passive parameters and optional Na/K ion channels) and attach receptors
(``AMPA``/``GABA``/``NMDA``/``AMPA_NMDA``) to chosen compartments. A
connection's ``receptor_type`` is the integer receptor index (add-order =
NEST ``syn_idx``).

This demo builds the **same two-compartment tree twice** — a soma (with active
Na/K channels) plus one dendrite — differing only in the dendrite:

* ``cm_pas`` — **passive** dendrite (leak only);
* ``cm_act`` — **active** dendrite (adds Na/K channels).

Both carry an ``AMPA_NMDA`` receptor on the soma (index 0) and on the dendrite
(index 1). Two ``spike_generator``s drive them identically:

1. ``sg_soma`` fires at 10/13/16 ms into the **somatic** receptor (weight 5 nS);
2. ``sg_dend`` fires at 70/73/76 ms into the **dendritic** receptor (weight 2 nS).

Both connections carry NEST's 0.5 ms delay. Watching the somatic and dendritic
voltages side by side shows how active dendritic channels (cm_act) amplify and
sharpen the dendritic response relative to the purely passive cable (cm_pas).

The live-NEST parity test (``brainpy_state/_nest/_validation/two_comps_test.py``)
drives a real ``cm_default`` with this identical wiring and confirms the per-
compartment ``v_comp`` / gating / receptor-conductance traces match.

Run:  PYTHONPATH=. python examples/nest_like/two_comps.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u

from brainpy.state import Simulator, cm_default, spike_generator, multimeter

DT = 0.1        # resolution [ms]
T_SIM = 160.0   # total simulation time [ms]

# Compartment parameters, verbatim from NEST two_comps.py (NEST-native units:
# pF, nS, mV). The soma always carries active Na/K channels.
SOMA_PARAMS = {
    'C_m': 89.245535, 'g_C': 0.0, 'g_L': 8.924572508, 'e_L': -75.0, 'v_comp': -75.0,
    'gbar_Na': 4608.698576715, 'e_Na': 60.0, 'gbar_K': 956.112772900, 'e_K': -90.0,
}
# Passive dendrite: leak + coupling only.
DEND_PASSIVE = {
    'C_m': 1.929929, 'g_C': 1.255439494, 'g_L': 0.192992878, 'e_L': -75.0, 'v_comp': -75.0,
}
# Active dendrite: same passive backbone (shifted leak) + Na/K channels.
DEND_ACTIVE = {
    'C_m': 1.929929, 'g_C': 1.255439494, 'g_L': 0.192992878, 'e_L': -70.0, 'v_comp': -70.0,
    'gbar_Na': 17.203212493, 'e_Na': 60.0, 'gbar_K': 11.887347450, 'e_K': -90.0,
}
V_TH = -50.0  # spike-detection threshold [mV]

# Receptor indices (add-order == NEST syn_idx): 0 on soma, 1 on dendrite.
SYN_SOMA = 0
SYN_DEND = 1

SG_SOMA_TIMES = [10.0, 13.0, 16.0]   # somatic spike-generator times [ms]
SG_DEND_TIMES = [70.0, 73.0, 76.0]   # dendritic spike-generator times [ms]
W_SOMA = 5.0   # somatic conductance increment per spike [nS]
W_DEND = 2.0   # dendritic conductance increment per spike [nS]
DELAY = 0.5    # connection delay [ms]

# The twelve recordables NEST's example plots: soma+dend voltage, soma+dend
# Na/K gating, and the dendritic AMPA_NMDA rise/decay conductance components.
RECORDABLES = ['v_comp0', 'v_comp1',
               'm_Na_0', 'h_Na_0', 'n_K_0',
               'm_Na_1', 'h_Na_1', 'n_K_1',
               'g_r_AN_AMPA_1', 'g_d_AN_AMPA_1',
               'g_r_AN_NMDA_1', 'g_d_AN_NMDA_1']


def _morphology(active_dend):
    """Return ``(compartments, receptors)`` for one two-compartment model.

    Parameters
    ----------
    active_dend : bool
        If ``True`` the dendrite carries Na/K channels (cm_act); otherwise it is
        a passive leak-only cable (cm_pas). The soma is always active.

    Returns
    -------
    compartments : list of dict
        ``[{'parent_idx': -1, 'params': soma}, {'parent_idx': 0, 'params': dend}]``.
    receptors : list of dict
        One ``AMPA_NMDA`` receptor on the soma (index 0) and one on the dendrite
        (index 1).
    """
    dend = DEND_ACTIVE if active_dend else DEND_PASSIVE
    compartments = [
        {'parent_idx': -1, 'params': dict(SOMA_PARAMS)},
        {'parent_idx': 0, 'params': dict(dend)},
    ]
    receptors = [
        {'comp_idx': 0, 'receptor_type': 'AMPA_NMDA'},   # index 0 (soma)
        {'comp_idx': 1, 'receptor_type': 'AMPA_NMDA'},   # index 1 (dendrite)
    ]
    return compartments, receptors


def build(simtime=T_SIM):
    """Build the Simulator with the passive- and active-dendrite models wired.

    Both models receive identical spike drive (``sg_soma`` → somatic receptor,
    ``sg_dend`` → dendritic receptor); only the dendritic compartment differs.

    Parameters
    ----------
    simtime : float, optional
        Simulation horizon in ms. Default :data:`T_SIM`.

    Returns
    -------
    sim : Simulator
    mm_pas, mm_act : NodeView
        Multimeters recording :data:`RECORDABLES` on the passive- and active-
        dendrite models (read via ``res.trace(mm, name)``).
    simtime : float
    """
    sim = Simulator(dt=DT * u.ms)

    comps_pas, rcpts_pas = _morphology(active_dend=False)
    comps_act, rcpts_act = _morphology(active_dend=True)
    cm_pas = sim.create(cm_default, 1,
                        params={'compartments': comps_pas, 'receptors': rcpts_pas, 'V_th': V_TH})
    cm_act = sim.create(cm_default, 1,
                        params={'compartments': comps_act, 'receptors': rcpts_act, 'V_th': V_TH})

    # Two spike_generators with fixed (deterministic) trains; each drives both
    # models identically, exactly as in NEST.
    sg_soma = sim.create(spike_generator, spike_times=np.asarray(SG_SOMA_TIMES) * u.ms)
    sg_dend = sim.create(spike_generator, spike_times=np.asarray(SG_DEND_TIMES) * u.ms)
    for cm in (cm_pas, cm_act):
        sim.connect(sg_soma, cm, receptor_type=SYN_SOMA, weight=W_SOMA * u.nS, delay=DELAY * u.ms)
        sim.connect(sg_dend, cm, receptor_type=SYN_DEND, weight=W_DEND * u.nS, delay=DELAY * u.ms)

    mm_pas = sim.create(multimeter, record_from=RECORDABLES, interval=DT * u.ms)
    mm_act = sim.create(multimeter, record_from=RECORDABLES, interval=DT * u.ms)
    sim.connect(mm_pas, cm_pas)
    sim.connect(mm_act, cm_act)
    return sim, mm_pas, mm_act, simtime


def _extract(res, mm):
    """Pull every recordable off one multimeter into a name→1-D-array dict.

    ``v_comp*`` traces are returned in mV; gating variables are dimensionless and
    conductance components are in nS (both returned as plain magnitudes).
    """
    traces = {}
    for name in RECORDABLES:
        tr = res.trace(mm, name)
        if name.startswith('v_comp'):
            tr = tr / u.mV
        traces[name] = np.asarray(u.get_mantissa(tr)).reshape(-1)
    return traces


def run_traces(simtime=T_SIM):
    """Run the demo and return ``(t_ms, traces)``.

    ``traces`` is ``{'pas': {...}, 'act': {...}}`` — one name→trace dict per model
    (passive-dendrite vs active-dendrite), each over :data:`RECORDABLES`.
    """
    sim, mm_pas, mm_act, _t = build(simtime)
    res = sim.simulate(simtime * u.ms)
    t = np.asarray(u.get_mantissa(res.times / u.ms))
    return t, {'pas': _extract(res, mm_pas), 'act': _extract(res, mm_act)}


def main():
    print("Two-compartment cm_default neurons: passive vs active dendrite (brainpy.state)")
    t, traces = run_traces()
    for key, label in (('pas', 'passive dend'), ('act', 'active dend')):
        tr = traces[key]
        vs, vd = tr['v_comp0'], tr['v_comp1']
        print(f"  {label:12s}: v_soma [{vs.min():7.2f}, {vs.max():7.2f}] mV, "
              f"v_dend [{vd.min():7.2f}, {vd.max():7.2f}] mV, "
              f"AMPA_dend max {(tr['g_r_AN_AMPA_1'] + tr['g_d_AN_AMPA_1']).max():.3f} nS")

    try:
        import matplotlib.pyplot as plt
        fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for key, col, label in (('pas', 'b', 'passive dend'), ('act', 'r', 'active dend')):
            ax_s.plot(t, traces[key]['v_comp0'], c=col, label=label)
            ax_d.plot(t, traces[key]['v_comp1'], c=col, label=label)
        ax_s.set_xlabel(r"$t$ (ms)"); ax_s.set_ylabel(r"$v_{soma}$ (mV)")
        ax_s.set_ylim(-90.0, 40.0); ax_s.legend(loc=0); ax_s.set_title("soma")
        ax_d.set_xlabel(r"$t$ (ms)"); ax_d.set_ylabel(r"$v_{dend}$ (mV)")
        ax_d.set_ylim(-90.0, 40.0); ax_d.legend(loc=0); ax_d.set_title("dendrite")
        fig.suptitle("cm_default: active vs passive dendrite")
        fig.tight_layout()
        fig.savefig("examples/nest_like/two_comps.png", dpi=100)
        print("  wrote examples/nest_like/two_comps.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
