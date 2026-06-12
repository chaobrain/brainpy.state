# examples/nest/precise_spiking.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Grid vs precise spiking — NEST-style port.

Port of NEST's ``precise_spiking.py`` (Morrison et al. 2007; Hanuschkin et al.
2010). The same constant DC current drives a grid-constrained
``iaf_psc_exp`` and its precise-spiking twin ``iaf_psc_exp_ps`` at several
resolutions; the demo contrasts their spike timing. The grid model can only
spike on the resolution grid, so its spike times shift as ``dt`` changes; the
precise model resolves the threshold crossing in continuous time, so its spike
times are (nearly) resolution-independent and land *between* grid points.

Two execution modes mirror the two device kinds:

* **Grid** (:func:`run_grid`) — built on the ``Simulator``: ``iaf_psc_exp`` +
  ``dc_generator`` + ``voltmeter`` + ``spike_recorder``, advanced by the JAX
  ``for_loop``. Spikes are read from ``res.spikes`` and the membrane trace from
  ``res.trace``.
* **Precise** (:func:`run_precise`) — driven **eagerly** in a plain Python loop
  (concrete ``t = k*dt``, 700 pA injected each step), mirroring
  ``iaf_psc_exp_ps_test.py``. The off-grid spike time is read from the neuron's
  ``last_spike_time`` after each firing. The precise integrator is event-driven
  in continuous time and is therefore kept outside the ``for_loop``.

Both neurons are deterministic, so live-NEST parity is exact up to a constant
onset shift (the DC connection-delay convention); see the parity test, which
aligns on the first spike.

Run:  python examples/nest/precise_spiking.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import saiunit as u

from brainpy_state import (Simulator, iaf_psc_exp, iaf_psc_exp_ps,
                           dc_generator, voltmeter, spike_recorder)

#: Injected DC amplitude (pA) — the upstream's suprathreshold drive.
STIM_CURRENT = 700.0
#: Simulation horizon (ms).
SIMTIME = 100.0
#: Resolutions (ms) swept by the demo.
RESOLUTIONS = (0.1, 0.5, 1.0)


def run_grid(dt, simtime=SIMTIME, stim_current=STIM_CURRENT):
    """Run the grid model ``iaf_psc_exp`` on the ``Simulator`` under a DC drive.

    Parameters
    ----------
    dt : float
        Simulation resolution in ms. The voltmeter samples on this grid.
    simtime : float, optional
        Horizon in ms. Default :data:`SIMTIME`.
    stim_current : float, optional
        DC amplitude in pA. Default :data:`STIM_CURRENT`.

    Returns
    -------
    dict
        ``{'spike_steps': int array of spike step indices,
        'vm': V_m trace (mV), 'times': sample times (ms)}``.
    """
    sim = Simulator(dt=dt * u.ms)
    neuron = sim.create(iaf_psc_exp, 1)
    dc = sim.create(dc_generator, amplitude=stim_current * u.pA)
    vm = sim.create(voltmeter, interval=dt * u.ms)
    sr = sim.create(spike_recorder)
    sim.connect(dc, neuron)
    sim.connect(vm, neuron)                  # reversed: the voltmeter observes the neuron
    sim.connect(neuron, sr)
    res = sim.simulate(simtime * u.ms)

    spk = np.asarray(res.spikes(sr)).reshape(-1)
    spike_steps = np.nonzero(spk > 0)[0].astype(int)
    vm_trace = np.asarray(u.get_mantissa(res.trace(vm, 'V_m') / u.mV)).reshape(-1)
    times = np.asarray(u.get_mantissa(res.times / u.ms)).reshape(-1)
    return {'spike_steps': spike_steps, 'vm': vm_trace, 'times': times}


def run_precise(dt, simtime=SIMTIME, stim_current=STIM_CURRENT):
    """Drive the precise model ``iaf_psc_exp_ps`` eagerly and collect spikes.

    The neuron is stepped in a plain Python loop with concrete ``t = k*dt`` and a
    constant ``stim_current`` injected each step. When a step fires, the off-grid
    spike time is read from ``last_spike_time``.

    Parameters
    ----------
    dt : float
        Simulation resolution in ms.
    simtime : float, optional
        Horizon in ms. Default :data:`SIMTIME`.
    stim_current : float, optional
        DC amplitude in pA. Default :data:`STIM_CURRENT`.

    Returns
    -------
    dict
        ``{'spike_steps': int array of grid step indices where firing was
        detected, 'spike_times': precise (off-grid) spike times (ms),
        'V_th': threshold (mV)}``.
    """
    n_steps = int(round(simtime / dt))
    spike_steps, spike_times = [], []
    with brainstate.environ.context(dt=dt * u.ms):
        neuron = iaf_psc_exp_ps(1)
        neuron.init_state()
        v_th = float(np.asarray(u.get_mantissa(neuron.V_th / u.mV)).reshape(-1)[0])
        for k in range(n_steps):
            with brainstate.environ.context(t=k * dt * u.ms):
                spk = neuron.update(x=stim_current * u.pA)
            if bool(u.math.all(spk > 0.0)):
                spike_steps.append(k)
                spike_times.append(
                    float((neuron.last_spike_time.value / u.ms).reshape(-1)[0]))
    return {'spike_steps': np.asarray(spike_steps, dtype=int),
            'spike_times': np.asarray(spike_times, dtype=float),
            'V_th': v_th}


def main():
    print("precise_spiking (brainpy.state, iaf_psc_exp grid vs iaf_psc_exp_ps precise)")
    print(f"  DC drive {STIM_CURRENT:.0f} pA, T={SIMTIME:.0f} ms")
    for dt in RESOLUTIONS:
        grid = run_grid(dt)
        prec = run_precise(dt)
        g_times = grid['spike_steps'] * dt
        print(f"  resolution {dt} ms:")
        if g_times.size and prec['spike_times'].size:
            print(f"    grid    iaf_psc_exp    : {g_times.size:2d} spikes, "
                  f"first @ {g_times[0]:7.3f} ms (on grid)")
            print(f"    precise iaf_psc_exp_ps : {prec['spike_times'].size:2d} spikes, "
                  f"first @ {prec['spike_times'][0]:7.4f} ms (off grid)")
        print(f"    V_m grid trace: {grid['vm'].shape[0]} samples, "
              f"min {grid['vm'].min():.2f} / max {grid['vm'].max():.2f} mV")
    print("  (grid spike times snap to the dt grid and shift with resolution; "
          "precise times resolve the crossing in continuous time)")


if __name__ == "__main__":
    main()
