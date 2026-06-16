# examples/nest_like/correlospinmatrix_detector_two_neuron.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Binary two-neuron correlations — NEST-style port.

Port of NEST's ``correlospinmatrix_detector_two_neuron.py`` (Ginzburg &
Sompolinsky, *Theory of correlations in stochastic neural networks*, 1994,
Fig. 1). A stochastic ``ginzburg_neuron`` (n1, Glauber dynamics, gain 0.5) drives
a deterministic ``mcculloch_pitts_neuron`` (n2, Heaviside threshold) with weight
1. A two-channel ``correlospinmatrix_detector`` records both binary trains and
yields the auto-/cross-covariance functions :math:`c_{11}, c_{12}, c_{21},
c_{22}` together with the per-channel mean activities.

**Coupling.** NEST delivers each n1 transition to n2 as a spike that shifts n2's
field ``h`` by :math:`\\pm w` (``+w`` on up, ``-w`` on down); these telescope to
:math:`h = w\\,y_1`. The Simulator's ``connect`` does not yet wire this binary
delta-coupling (``network-api-gap.md``), so it is reproduced directly inside a
pure-JAX ``for_loop``: each step drives n2 with n1's **one-step-delayed** state
``x = w * y1_prev`` (a current input, identical at the gain evaluation
:math:`H(w\\,y_1 - \\theta)` to NEST's telescoped ``h``). The neurons run *inside*
the loop; nothing imperative does.

**Recording.** ``correlospinmatrix_detector`` is an imperative host device (event
deques, Python loops), so it is fed **eagerly** (post-hoc) from the recorded
trains. Each binary transition is encoded as NEST delivers it — an up-transition
(0->1) as **two** unit spikes at one stamp, a down-transition (1->0) as **one**.
Only transition steps are fed (an empty ``update()`` is a no-op), so the cost is
``O(n_transitions)``. The detector itself mirrors NEST bit-for-bit (see
``correlospinmatrix_detector_test.py``); this module only reproduces the wiring.

Because the ginzburg and mcculloch-pitts PRNG streams diverge from NEST's, parity
is distributional (seed-mean mean activities + seed-averaged covariance
functions); see the parity test.

Run:  python examples/nest_like/correlospinmatrix_detector_two_neuron.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

import numpy as np
import brainunit as u

from brainpy.state import (ginzburg_neuron, mcculloch_pitts_neuron,
                           correlospinmatrix_detector)

#: Target mean activity of the autonomous neuron n1 (gain ``2*m_x`` -> 0.5).
M_X = 0.5
#: Membrane time constant (ms) of both binary neurons.
TAU_M = 10.0
#: Simulation resolution (ms).
RESOLUTION = 0.1
#: One-sided correlation lag horizon (ms).
TAU_MAX = 100.0
#: Detector lag-bin width (ms) — the upstream uses the resolution itself.
DELTA_TAU = 0.1
#: Synaptic weight of the n1 -> n2 projection.
WEIGHT = 1.0


def simulate_pair(seed, simtime, dt=RESOLUTION, weight=WEIGHT, tau_m=TAU_M, m_x=M_X):
    """Run the coupled binary pair and return their recorded trains.

    The two neurons advance together in a single JAX ``for_loop``. n2 reads n1's
    state from the *previous* step (``n1.y`` before n1 updates), reproducing the
    one-step synaptic latency of NEST's transition delivery; the drive
    ``x = weight * y1_prev`` is the telescoped analog of NEST's ``h = w * y1``.

    Parameters
    ----------
    seed : int
        PRNG seed; n1 uses ``seed``, n2 uses ``seed + 500`` (decorrelated streams).
    simtime : float
        Simulation horizon in ms.
    dt : float, optional
        Resolution in ms. Default :data:`RESOLUTION`.
    weight : float, optional
        n1 -> n2 synaptic weight. Default :data:`WEIGHT`.
    tau_m : float, optional
        Membrane time constant in ms for both neurons. Default :data:`TAU_M`.
    m_x : float, optional
        Target activity of n1 (sets its gain ``c_2 = 2 * m_x``). Default :data:`M_X`.

    Returns
    -------
    y1, y2 : numpy.ndarray
        Binary trains (``0.0``/``1.0``) of shape ``(n_steps,)`` for n1 and n2.
    """
    n_steps = int(round(simtime / dt))
    dt_q = dt * u.ms
    times = u.math.arange(n_steps) * dt_q

    brainstate.random.seed(seed + 1000)
    with brainstate.environ.context(dt=dt_q):
        n1 = ginzburg_neuron(1, theta=0. * u.mV, tau_m=tau_m * u.ms,
                             c_1=0. / u.mV, c_2=2.0 * m_x, c_3=1.0 / u.mV,
                             stochastic_update=True, rng_seed=seed)
        n2 = mcculloch_pitts_neuron(1, theta=0.5 * u.mV, tau_m=tau_m * u.ms,
                                    stochastic_update=True, rng_seed=seed + 500)
        brainstate.nn.init_all_states(n1)
        brainstate.nn.init_all_states(n2)

        def step(t):
            with brainstate.environ.context(t=t):
                y1_prev = n1.y.value                       # n1 state from the previous step
                y2 = n2.update(x=weight * y1_prev * u.mV)  # one-step-delayed drive
                y1 = n1.update()
            return u.math.concatenate([y1, y2])

        ys = np.asarray(brainstate.transform.for_loop(step, times))
    return ys[:, 0].copy(), ys[:, 1].copy()


def _transitions(train):
    """Up/down transition step indices of a binary train (0->1 and 1->0)."""
    train = np.asarray(train)
    dy = np.diff(np.concatenate([[0.0], train]))
    return np.nonzero(dy > 0.5)[0], np.nonzero(dy < -0.5)[0]


def run_correlospinmatrix(y1, y2, dt=RESOLUTION, delta_tau=DELTA_TAU,
                          tau_max=TAU_MAX, Tstart=TAU_MAX):
    """Feed the two binary trains to the eager ``correlospinmatrix_detector``.

    Transitions are encoded exactly as NEST's binary neurons emit them — an
    up-transition as **two** unit spikes sharing one stamp, a down-transition as
    **one** — and stamped at ``step + 1`` (NEST's one-step delivery latency).
    Only transition steps are fed; events on the same stamp are ordered by
    channel.

    Parameters
    ----------
    y1, y2 : array_like
        Binary trains of equal length for receptor channels 0 and 1.
    dt : float, optional
        Resolution in ms. Default :data:`RESOLUTION`.
    delta_tau, tau_max : float, optional
        Detector lag-bin width and one-sided horizon in ms. Defaults
        :data:`DELTA_TAU`, :data:`TAU_MAX`.
    Tstart : float, optional
        NEST ``Tstart`` (ms), forwarded for API fidelity. Default :data:`TAU_MAX`.

    Returns
    -------
    numpy.ndarray
        Raw ``count_covariance`` tensor of shape
        ``(2, 2, 1 + 2 * tau_max / delta_tau)`` (overlap durations in steps).
    """
    dt_q = dt * u.ms
    events = {}
    for channel, train in ((0, y1), (1, y2)):
        ups, downs = _transitions(train)
        for k in ups:
            events.setdefault(int(k) + 1, []).extend([(channel, 1), (channel, 1)])
        for k in downs:
            events.setdefault(int(k) + 1, []).append((channel, 1))

    with brainstate.environ.context(dt=dt_q):
        cmd = correlospinmatrix_detector(
            N_channels=2, tau_max=tau_max * u.ms,
            Tstart=Tstart * u.ms, delta_tau=delta_tau * u.ms)
        for stamp in sorted(events):
            evs = sorted(events[stamp], key=lambda e: e[0])
            k = len(evs)
            with brainstate.environ.context(t=(stamp - 1) * dt_q):
                cmd.update(
                    spikes=np.ones((k,)),
                    receptor_ports=np.asarray([e[0] for e in evs]),
                    multiplicities=np.asarray([e[1] for e in evs]),
                    stamp_steps=np.full((k,), stamp),
                )
        return np.asarray(cmd.get('count_covariance')).astype(float)


def mean_activities(count_covariance, dt=RESOLUTION, simtime=None):
    """Per-channel mean activity from the zero-lag auto-covariance.

    Mirrors the upstream ``count_covariance[i][i][tau_max/h] * h / T``: the
    zero-lag auto-covariance bin holds the total time each channel spent in the
    up state (in steps), so scaling by ``dt / simtime`` gives the occupancy.

    Parameters
    ----------
    count_covariance : array_like
        Raw ``(2, 2, n_bins)`` detector tensor from :func:`run_correlospinmatrix`.
    dt : float, optional
        Resolution in ms. Default :data:`RESOLUTION`.
    simtime : float
        Simulation horizon in ms (``T``). Required.

    Returns
    -------
    numpy.ndarray
        Mean activities ``(2,)`` for channels 0 and 1.
    """
    if simtime is None:
        raise ValueError("simtime (T) is required.")
    cc = np.asarray(count_covariance, dtype=float)
    center = cc.shape[-1] // 2                       # zero-lag bin
    return np.array([cc[i, i, center] * (dt / simtime) for i in range(2)])


def covariance_matrix(count_covariance, mean_act, dt=RESOLUTION, simtime=None):
    """Centered auto-/cross-covariance functions (upstream formula).

    ``covariance[i, j] = count_covariance[i][j] * h / T - m_i * m_j``.

    Parameters
    ----------
    count_covariance : array_like
        Raw ``(2, 2, n_bins)`` detector tensor.
    mean_act : array_like
        Per-channel mean activities ``(2,)`` from :func:`mean_activities`.
    dt : float, optional
        Resolution in ms. Default :data:`RESOLUTION`.
    simtime : float
        Simulation horizon in ms (``T``). Required.

    Returns
    -------
    numpy.ndarray
        Covariance functions ``(2, 2, n_bins)`` over lags.
    """
    if simtime is None:
        raise ValueError("simtime (T) is required.")
    cc = np.asarray(count_covariance, dtype=float)
    m = np.asarray(mean_act, dtype=float)
    n_bins = cc.shape[-1]
    cov = np.zeros((2, 2, n_bins))
    for i in range(2):
        for j in range(2):
            cov[i, j] = cc[i, j] * (dt / simtime) - m[i] * m[j]
    return cov


def main(seed=0, simtime=50000.0):
    y1, y2 = simulate_pair(seed=seed, simtime=simtime)
    cc = run_correlospinmatrix(y1, y2)
    ma = mean_activities(cc, simtime=simtime)
    cov = covariance_matrix(cc, ma, simtime=simtime)
    center = cc.shape[-1] // 2

    print("correlospinmatrix_detector_two_neuron (brainpy.state, ginzburg -> mcculloch_pitts)")
    print(f"  weight={WEIGHT}, tau_m={TAU_M} ms, T={simtime:.0f} ms, "
          f"tau_max={TAU_MAX} ms, delta_tau={DELTA_TAU} ms")
    print(f"  direct train means: n1={y1.mean():.4f}, n2={y2.mean():.4f}")
    print(f"  detector mean activities: n1={ma[0]:.4f}, n2={ma[1]:.4f}")
    print("  covariance peaks (zero lag):")
    print(f"    c11={cov[0, 0][center]:.4f}  c22={cov[1, 1][center]:.4f}  "
          f"(auto, ~ m*(1-m))")
    print(f"    c12={cov[0, 1].max():.4f}  c21={cov[1, 0].max():.4f}  (cross peak)")
    print("  (n1->n2 binary coupling done in-loop: network-api-gap.md — "
          "Simulator.connect lacks binary delta-coupling)")


if __name__ == "__main__":
    main()
