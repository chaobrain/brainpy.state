# examples/nest/lin_rate_ipn_network.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Network of linear rate neurons -- NEST ``lin_rate_ipn_network.py`` port.

An excitatory (``NE = 4*order``) and an inhibitory (``NI = order``) population of
``lin_rate_ipn`` neurons with **delayed excitatory** and **instantaneous inhibitory**
connections, relaxed end-to-end through the :class:`~brainpy.state.Simulator` (one
compiled ``for_loop`` -- no Python step loop). The per-step ``rate`` of one excitatory and
one inhibitory neuron is plotted, reproducing NEST's tutorial figure.

Connections originating from excitatory neurons carry a delay ``d_e`` (NEST's
``rate_connection_delayed``); connections originating from inhibitory neurons are
instantaneous (``rate_connection_instantaneous``). The substrate realises the
instantaneous coupling as a plain ``connect(weight=..., comm='dense')`` and the delayed
coupling by adding ``delay=...``; its one-step pipeline lag is exactly NEST's
``use_wfr=False`` instantaneous seed.

NEST wires the populations with ``fixed_outdegree``; brainpy.state exposes
``fixed_indegree`` (there is no ``fixed_outdegree`` rule), so each connection's outdegree
``K_out`` is mapped to the mean-field-equivalent indegree ``K_in = N_src * K_out / N_tgt``
(identical expected in-degree, hence the same mean-field input). With ``sigma > 0`` the
linear network fluctuates about its mean fixed point ``r* = (lambda I - W)^{-1} mu`` -- the
deterministic fixed point asserted tightly against NEST in
``brainpy_state/_nest/_validation/lin_rate_ipn_network_test.py``.

Run:  PYTHONPATH=. python examples/nest/lin_rate_ipn_network.py
"""
import jax
import brainstate

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import brainunit as u

from brainpy import state as bp


def run(order=50, T=100.0, dt=0.1, g=5.0, epsilon=0.1, d_e=5.0, sigma=5.0, seed=0):
    r"""Build and relax the excitatory/inhibitory linear-rate network.

    Parameters
    ----------
    order : int, optional
        Population scale: ``NE = 4 * order`` excitatory and ``NI = order`` inhibitory
        ``lin_rate_ipn`` neurons. Default ``50`` (NEST's value).
    T : float, optional
        Simulation time in ms. Default ``100``.
    dt : float, optional
        Time resolution in ms. Default ``0.1``.
    g : float, optional
        Inhibitory/excitatory weight ratio. Default ``5``.
    epsilon : float, optional
        Connection density, setting the per-population degrees. Default ``0.1``.
    d_e : float, optional
        Delay of the (delayed) excitatory connections in ms. Default ``5``.
    sigma : float, optional
        Input-noise scale of every ``lin_rate_ipn``. Default ``5``.
    seed : int, optional
        Seed for connectivity sampling and the input noise. Default ``0``.

    Returns
    -------
    rate_e0 : numpy.ndarray
        ``rate`` trace of the first excitatory neuron, shape ``(n_steps,)``.
    rate_i0 : numpy.ndarray
        ``rate`` trace of the first inhibitory neuron, shape ``(n_steps,)``.
    times : numpy.ndarray
        Time axis in ms, shape ``(n_steps,)``.
    rate_e : numpy.ndarray
        Full excitatory rate trace, shape ``(n_steps, NE)``.
    rate_i : numpy.ndarray
        Full inhibitory rate trace, shape ``(n_steps, NI)``.

    Examples
    --------
    .. code-block:: python

       >>> rate_e0, rate_i0, times, rate_e, rate_i = run(order=5, T=20.0)
       >>> bool(np.all(np.isfinite(rate_e0)) and rate_e.shape[1] == 20)
       True
    """
    brainstate.random.seed(seed)
    NE, NI = 4 * order, order
    N = NE + NI
    w = 0.1 / np.sqrt(N)
    KE, KI = int(epsilon * NE), int(epsilon * NI)
    # outdegree -> indegree (mean-field-equivalent): connections TO E use the KE degree,
    # connections TO I use the KI degree; guard a minimum of one edge for tiny ``order``.
    indeg_E_from_E = max(1, round(NE * KE / NE))   # each E gets ~KE excitatory inputs
    indeg_I_from_E = max(1, round(NE * KI / NI))   # each I gets ~NE*KI/NI excitatory inputs
    indeg_E_from_I = max(1, round(NI * KE / NE))   # each E gets ~NI*KE/NE inhibitory inputs
    indeg_I_from_I = max(1, round(NI * KI / NI))   # each I gets ~KI inhibitory inputs

    npar = dict(tau=10.0 * u.ms, mu=2.0, sigma=sigma, lambda_=1.0, g=1.0,
                linear_summation=True)
    net = bp.Simulator(dt=dt * u.ms)
    n_e = net.create(bp.lin_rate_ipn, NE, params=npar)
    n_i = net.create(bp.lin_rate_ipn, NI, params=npar)

    # Excitatory-origin connections are delayed; inhibitory-origin are instantaneous.
    net.connect(n_e, n_e, weight=w, delay=d_e * u.ms,
                rule=bp.fixed_indegree(indeg_E_from_E), comm='dense', seed=seed)
    net.connect(n_e, n_i, weight=w, delay=d_e * u.ms,
                rule=bp.fixed_indegree(indeg_I_from_E), comm='dense', seed=seed + 1)
    net.connect(n_i, n_e, weight=-g * w,
                rule=bp.fixed_indegree(indeg_E_from_I), comm='dense', seed=seed + 2)
    net.connect(n_i, n_i, weight=-g * w,
                rule=bp.fixed_indegree(indeg_I_from_I), comm='dense', seed=seed + 3)

    mm_e = net.create(bp.multimeter, record_from=['rate'])
    mm_i = net.create(bp.multimeter, record_from=['rate'])
    net.connect(mm_e, n_e)
    net.connect(mm_i, n_i)

    res = net.simulate(T * u.ms)
    rate_e = np.asarray(u.get_mantissa(res.trace(mm_e, 'rate'))).reshape(-1, NE)
    rate_i = np.asarray(u.get_mantissa(res.trace(mm_i, 'rate'))).reshape(-1, NI)
    times = (np.arange(rate_e.shape[0]) + 1) * dt
    return rate_e[:, 0], rate_i[:, 0], times, rate_e, rate_i


def main():  # pragma: no cover
    rate_e0, rate_i0, times, _, _ = run()
    print('lin_rate_ipn_network (brainpy.state)')
    print(f'  final excitatory rate : {rate_e0[-1]:.3f}')
    print(f'  final inhibitory rate : {rate_i0[-1]:.3f}')
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(times, rate_e0, label='excitatory')
        plt.plot(times, rate_i0, label='inhibitory')
        plt.xlabel('time (ms)')
        plt.ylabel('rate (a.u.)')
        plt.title('Network of linear rate neurons (brainpy.state)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('examples/nest/lin_rate_ipn_network_rates.png', dpi=100)
        print('  wrote examples/nest/lin_rate_ipn_network_rates.png')
    except ImportError:
        print('  (matplotlib not installed; skipping plot)')


if __name__ == '__main__':
    main()
