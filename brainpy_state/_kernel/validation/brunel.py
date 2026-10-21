# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Brunel (2000) model A with alpha PSCs on the K0 kernel.

Same network as ``benchmarks/brunel_simulator_baseline.py`` (NEST
``brunel_alpha_nest.py`` parameters): ``N_E = 4 * order`` and ``N_I = order``
``iaf_psc_alpha`` neurons, ``fixed_indegree`` recurrence (``C_E = 0.1 N_E``,
``C_I = 0.1 N_I``) from ``E`` and ``I`` onto the concatenated view ``E + I``, delay
1.5 ms, and one independent Poisson train per neuron (NEST draws per target, so the
kernel uses one ``poisson_generator`` unit per neuron, ``one_to_one``).
"""
from __future__ import annotations

import math
from typing import Any

import brainunit as u
import numpy as np

from brainpy_state._kernel.connectivity import fixed_indegree, one_to_one
from brainpy_state._kernel.models import iaf_psc_alpha, poisson_generator
from brainpy_state._kernel.network import Network

__all__ = ['psp_norm', 'brunel_params', 'build_brunel', 'rates']


def psp_norm(tau_mem: float, c_mem: float, tau_syn: float) -> float:
    """PSC amplitude (pA) giving a 1 mV PSP peak for the alpha kernel (NEST example)."""
    import scipy.special as sp
    a = tau_mem / tau_syn
    b = 1.0 / tau_syn - 1.0 / tau_mem
    x = -math.exp(-1.0 / a) / a
    lw = sp.lambertw(x, k=-1 if x < 0 else 0).real
    t_max = 1.0 / b * (-lw - 1.0 / a)
    return (math.exp(1.0) / (tau_syn * c_mem * b)
            * ((math.exp(-t_max / tau_mem) - math.exp(-t_max / tau_syn)) / b
               - t_max * math.exp(-t_max / tau_syn)))


def brunel_params(order: int, g: float = 5.0, eta: float = 2.0, epsilon: float = 0.1) -> dict:
    """Derived parameters of the network (sizes, in-degrees, weights in pA, Poisson rate in Hz)."""
    ne, ni = 4 * order, order
    ce, ci = int(epsilon * ne), int(epsilon * ni)
    tau_syn, tau_mem, c_mem, theta = 0.5, 20.0, 250.0, 20.0
    j_ex = 0.1 / psp_norm(tau_mem, c_mem, tau_syn)
    nu_th = (theta * c_mem) / (j_ex * ce * math.exp(1) * tau_mem * tau_syn)
    return dict(N_E=ne, N_I=ni, C_E=ce, C_I=ci, J_ex=j_ex, J_in=-g * j_ex,
                rate=1000.0 * eta * nu_th * ce, tau_syn=tau_syn, tau_mem=tau_mem, c_mem=c_mem,
                theta=theta, t_ref=2.0, delay=1.5)


def build_brunel(order: int, *, dt: float = 0.1, seed: int = 0, storage: str = 'auto',
                 n_rec: int = 50, record_v: bool = False,
                 delay_buffer: str = 'auto') -> tuple[Network, dict]:
    """Build the network.

    Parameters
    ----------
    order : int
        Network scale (``N = 5 * order``).
    dt : float, default 0.1
        Step in ms.
    seed : int, default 0
        Network seed.
    storage : {'auto', 'dense', 'csr'}
        Storage of the recurrent projections.
    n_rec : int, default 50
        Recorded neurons per population (spikes).
    record_v : bool, default False
        Also record ``V_m`` of the recorded E neurons.
    delay_buffer : {'auto', 'pre', 'post'}, default 'auto'
        Delay-buffer placement of every projection.

    Returns
    -------
    net : Network
    info : dict
        ``params`` (see :func:`brunel_params`), populations ``E``, ``I``, ``noise``,
        observers ``rec_E``, ``rec_I`` (and ``rec_V``).

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._kernel.validation.brunel import build_brunel, rates
        >>> net, info = build_brunel(20, seed=1)
        >>> res = net.compile().runner().simulate(200.0 * u.ms)
        >>> e_rate, i_rate = rates(res, info)
        >>> 10.0 < e_rate < 150.0
        True
    """
    p = brunel_params(order)
    npar = dict(C_m=p['c_mem'] * u.pF, tau_m=p['tau_mem'] * u.ms, tau_syn_ex=p['tau_syn'] * u.ms,
                tau_syn_in=p['tau_syn'] * u.ms, t_ref=p['t_ref'] * u.ms, E_L=0.0 * u.mV,
                V_reset=0.0 * u.mV, V_th=p['theta'] * u.mV)
    net = Network(dt=dt * u.ms, seed=seed)
    E = net.add(iaf_psc_alpha, p['N_E'], params=npar, name='E')
    I = net.add(iaf_psc_alpha, p['N_I'], params=npar, name='I')
    noise = net.add(poisson_generator, p['N_E'] + p['N_I'], params=dict(rate=p['rate'] * u.Hz),
                    name='noise')
    d = p['delay'] * u.ms
    kw = dict(delay=d, delay_buffer=delay_buffer)
    net.connect(noise, E + I, one_to_one(), port='ex', weight=p['J_ex'] * u.pA, name='noise_EI', **kw)
    net.connect(E, E + I, fixed_indegree(p['C_E']), port='ex', weight=p['J_ex'] * u.pA,
                storage=storage, name='E_EI', **kw)
    net.connect(I, E + I, fixed_indegree(p['C_I']), port='in_', weight=p['J_in'] * u.pA,
                storage=storage, name='I_EI', **kw)
    k = min(n_rec, p['N_E'])
    info: dict[str, Any] = dict(params=p, E=E, I=I, noise=noise,
                                rec_E=net.observe(E[:k], 'spike', name='rec_E'),
                                rec_I=net.observe(I[:min(n_rec, p['N_I'])], 'spike', name='rec_I'))
    if record_v:
        info['rec_V'] = net.observe(E[:k], 'V_m', name='rec_V')
    return net, info


def rates(res: Any, info: dict) -> tuple[float, float]:
    """Mean E and I rates (Hz) of the recorded neurons in a ``Results`` (unbatched)."""
    dur_s = float(u.Quantity(res.t_stop - res.t_start).to_decimal(u.second))
    out = []
    for key in ('rec_E', 'rec_I'):
        x = np.asarray(res[info[key]]['spike'])
        out.append(float(x.sum(axis=-2).mean()) / dur_s if x.size and dur_s > 0 else 0.0)
    return out[0], out[1]
