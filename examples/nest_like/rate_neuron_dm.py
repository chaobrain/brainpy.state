# examples/nest_like/rate_neuron_dm.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Rate-neuron decision making -- NEST ``rate_neuron_dm.py`` port.

Two ``lin_rate_ipn`` units (``lambda = 0.1``, ``tau = 1 ms``, ``rectify_output=True``) in
mutual instantaneous inhibition (``weight = -0.2``) form a winner-take-all decision circuit.
Evidence for each choice is the unit's mean input ``mu``; because the coupling matrix
``lambda*I - W`` is indefinite the rectified dynamics are bistable: the winner relaxes to
``mu_win / lambda = 10 * mu_win`` and the loser is rectified to ``0``. A positive evidence
bias ``dE`` selects the higher-``mu`` unit; with ``dE = 0`` the input noise breaks the tie.

Each scenario runs a no-evidence phase (``mu = 0``) then an evidence phase
(``mu = 1 +/- dE``), each relaxed end-to-end through the
:class:`~brainpy.state.Simulator` (one compiled ``for_loop`` per phase -- no Python step
loop). The :class:`~brainpy.state.Simulator` re-initialises state at the start of every
:meth:`~brainpy.state.Simulator.simulate`, so the two-phase protocol is realised as two
independent relaxations whose evidence phase starts from ``rate = 0`` -- numerically the
same state NEST holds at the end of its (rate-0) no-evidence phase, so the decision matches.

Run:  PYTHONPATH=. python examples/nest_like/rate_neuron_dm.py
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy import state as bp

#: Mutual inhibitory weight between the two decision units (NEST's value).
W_INH = -0.2


def build_network(sigma, dt, mu_d1, mu_d2):
    """Two mutually-inhibiting ``lin_rate_ipn`` decision units and their recorders.

    Parameters
    ----------
    sigma : float
        Input-noise scale of both units (``0`` for the deterministic regime).
    dt : float
        Integration step in ms.
    mu_d1, mu_d2 : float
        Mean input (evidence) of unit D1 / D2 for this phase.

    Returns
    -------
    net : brainpy.state.Simulator
        The wired two-unit decision network.
    mm1, mm2 : brainpy.state.NodeView
        Multimeters recording the ``rate`` of D1 and D2.
    """
    common = dict(tau=1.0 * u.ms, lambda_=0.1, sigma=sigma,
                  rectify_output=True, linear_summation=True)
    net = bp.Simulator(dt=dt * u.ms)
    d1 = net.create(bp.lin_rate_ipn, 1, params=dict(mu=mu_d1, **common))
    d2 = net.create(bp.lin_rate_ipn, 1, params=dict(mu=mu_d2, **common))
    # Mutual instantaneous inhibition (plain weighted dense connect, no delay).
    net.connect(d1, d2, weight=W_INH, comm='dense')
    net.connect(d2, d1, weight=W_INH, comm='dense')
    mm1 = net.create(bp.multimeter, record_from=['rate'])
    mm2 = net.create(bp.multimeter, record_from=['rate'])
    net.connect(mm1, d1)
    net.connect(mm2, d2)
    return net, mm1, mm2


def run_scenario(sigma, dE, dt=0.1, T_each=100.0, seed=0):
    """Run the two-phase decision protocol; return ``(rate_d1, rate_d2, times)``.

    Phase 1 presents no evidence (``mu = 0``); phase 2 presents evidence
    ``mu = 1 +/- dE``. Both phases are relaxed through the
    :class:`~brainpy.state.Simulator` and their ``rate`` traces concatenated.

    Parameters
    ----------
    sigma : float
        Input-noise scale of both decision units.
    dE : float
        Evidence bias: D1 receives ``1 + dE``, D2 receives ``1 - dE`` in phase 2.
    dt : float, optional
        Integration step in ms. Default ``0.1``.
    T_each : float, optional
        Duration of *each* phase in ms. Default ``100``.
    seed : int, optional
        Seed for the input noise. Default ``0``.

    Returns
    -------
    rate_d1, rate_d2 : numpy.ndarray
        Concatenated (phase 1 then phase 2) ``rate`` traces of D1 / D2,
        shape ``(2 * n_steps_per_phase,)``.
    times : numpy.ndarray
        Time axis in ms, shape matching the rate traces.

    Examples
    --------
    .. code-block:: python

       >>> r1, r2, t = run_scenario(sigma=0.0, dE=0.1, T_each=50.0)
       >>> bool(r1[-1] > r2[-1])      # positive bias -> D1 wins
       True
    """
    brainstate.random.seed(seed)
    # Phase 1: no evidence. Phase 2: evidence mu = 1 +/- dE. simulate() re-inits state,
    # so each phase relaxes from rate = 0 while sharing one RNG stream across the run.
    net1, mm1a, mm2a = build_network(sigma, dt, mu_d1=0.0, mu_d2=0.0)
    res1 = net1.simulate(T_each * u.ms)
    net2, mm1b, mm2b = build_network(sigma, dt, mu_d1=1.0 + dE, mu_d2=1.0 - dE)
    res2 = net2.simulate(T_each * u.ms)

    def _cat(res_a, mm_a, res_b, mm_b):
        a = np.asarray(u.get_mantissa(res_a.trace(mm_a, 'rate'))).reshape(-1)
        b = np.asarray(u.get_mantissa(res_b.trace(mm_b, 'rate'))).reshape(-1)
        return np.concatenate([a, b])

    r1 = _cat(res1, mm1a, res2, mm1b)
    r2 = _cat(res1, mm2a, res2, mm2b)
    times = (np.arange(r1.shape[0]) + 1) * dt
    return r1, r2, times


def main():  # pragma: no cover
    sigmas, dEs = [0.0, 0.1, 0.2], [0.0, 0.004, 0.008]
    print('rate_neuron_dm (brainpy.state) -- WTA decision grid')
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=True)
        for r, sigma in enumerate(sigmas):
            for c, dE in enumerate(dEs):
                r1, r2, t = run_scenario(sigma, dE, T_each=100.0, seed=r * 3 + c)
                ax = axes[r, c]
                ax.plot(t, r1, 'b', label='D1')
                ax.plot(t, r2, 'r', label='D2')
                ax.set_ylim([-0.5, 12.0])
                if c == 0:
                    ax.set_ylabel(f'activity (sigma={sigma})')
                if r == 0:
                    ax.set_title(f'dE={dE}')
                if r == 2:
                    ax.set_xlabel('time (ms)')
        axes[0, 2].legend()
        plt.suptitle('Rate-neuron decision making (brainpy.state)')
        plt.tight_layout()
        plt.savefig('examples/nest_like/rate_neuron_dm.png', dpi=100)
        print('  wrote examples/nest_like/rate_neuron_dm.png')
    except ImportError:
        r1, r2, _ = run_scenario(0.0, 0.008, T_each=100.0)
        print(f'  dE=0.008 -> D1={r1[-1]:.3f}, D2={r2[-1]:.3f} '
              '(matplotlib absent; skipping grid plot)')


if __name__ == '__main__':
    main()
