# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""COBA network (Vogels & Abbott 2005; Brian2's ``COBA`` benchmark) on the kernel and in Brian2.

4000 conductance-based neurons (3200 E, 800 I) with ``iaf_cond_exp`` dynamics,
``pairwise_bernoulli(0.02)`` without autapses, synaptic delay one step, constant
drive ``I_e = 200 pA`` (Brian2's ``I = 20 mV`` times ``g_L``), initial
``V ~ U(V_reset, V_th)``. Gate G3 compares the network statistics of both simulators
and single-neuron traces driven by identical input spikes.

The Brian2 model is written to be the same dynamical system as NEST's / the
kernel's ``iaf_cond_exp``:

* the voltage in the vector field is bounded, ``min(V, V_th)`` (NEST);
* the threshold is checked after the step and ``V`` is clamped at ``V_reset`` for
  ``round_up(t_ref / dt)`` steps after the spike step: Brian2 counts the spike step
  into ``refractory``, so it gets ``t_ref + dt``;
* Brian2 stamps a spike with the start of its step (``n dt``), NEST / the kernel with
  its end (``(n + 1) dt``): Brian2 spike times are shifted by ``+dt``;
* recurrent synapses have ``delay = dt`` (the kernel needs at least one step on a
  cycle; both then add the jump after the integration of the arrival step); a
  ``SpikeGeneratorGroup`` input at ``T`` uses Brian2 ``delay = 0`` against the kernel
  ``spike_generator`` (stamp ``T``, emitted in step ``T/dt - 1``) with delay ``dt``.
"""
from __future__ import annotations

from typing import Any

import brainunit as u
import numpy as np

from brainpy_state._kernel.connectivity import pairwise_bernoulli
from brainpy_state._kernel.models import iaf_cond_exp, spike_generator
from brainpy_state._kernel.network import Network
from brainpy_state._kernel.params import Uniform

__all__ = ['COBA', 'build_coba', 'kernel_coba', 'brian2_coba', 'kernel_single', 'brian2_single']

#: Parameters (mantissas: ms, mV, pF, nS, pA).
COBA = dict(N_E=3200, N_I=800, p=0.02, dt=0.1, C_m=200.0, g_L=10.0, E_L=-60.0,
            V_reset=-60.0, V_th=-50.0, t_ref=5.0, I_e=200.0, tau_ex=5.0, tau_in=10.0,
            E_ex=0.0, E_in=-80.0, w_ex=6.0, w_in=67.0)


def _neuron_params(c: dict) -> dict:
    return dict(C_m=c['C_m'] * u.pF, g_L=c['g_L'] * u.nS, E_L=c['E_L'] * u.mV,
                V_reset=c['V_reset'] * u.mV, V_th=c['V_th'] * u.mV, t_ref=c['t_ref'] * u.ms,
                I_e=c['I_e'] * u.pA, tau_syn_ex=c['tau_ex'] * u.ms, tau_syn_in=c['tau_in'] * u.ms,
                E_ex=c['E_ex'] * u.mV, E_in=c['E_in'] * u.mV)


def build_coba(seed: int = 0, scale: float = 1.0) -> tuple[Network, dict]:
    """Kernel COBA network; ``scale`` multiplies both population sizes (tests use < 1).

    Returns
    -------
    net : Network
    info : dict
        ``P`` (the whole population, E first), ``rec`` (spike observer of all neurons),
        ``N_E``, ``N_I``.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._kernel.validation.coba import build_coba
        >>> net, info = build_coba(seed=1, scale=0.05)
        >>> res = net.compile().runner().simulate(50.0 * u.ms)
        >>> res[info['rec']]['spike'].shape
        (500, 200)
    """
    c = COBA
    ne, ni = int(round(c['N_E'] * scale)), int(round(c['N_I'] * scale))
    net = Network(dt=c['dt'] * u.ms, seed=seed)
    P = net.add(iaf_cond_exp, ne + ni, params=_neuron_params(c),
                init=dict(V_m=Uniform(c['V_reset'] * u.mV, c['V_th'] * u.mV)), name='P')
    d = c['dt'] * u.ms
    rule = pairwise_bernoulli(c['p'], allow_autapses=False)
    net.connect(P[:ne], P, rule, port='ex', weight=c['w_ex'] * u.nS, delay=d, name='E_P')
    net.connect(P[ne:], P, rule, port='in_', weight=c['w_in'] * u.nS, delay=d, name='I_P')
    rec = net.observe(P, 'spike', name='rec')
    return net, dict(P=P, rec=rec, N_E=ne, N_I=ni)


def kernel_coba(seed: int, t_ms: float, precision: str = 'float32', scale: float = 1.0):
    """Spike events ``(times ms, senders)`` of one kernel COBA run (senders E first)."""
    net, info = build_coba(seed, scale)
    res = net.compile(precision).runner().simulate(t_ms * u.ms)
    t, gids, counts = res[info['rec']].spike_events()
    counts = np.asarray(counts, int)
    return (np.repeat(np.asarray(t.to_decimal(u.ms)), counts),
            np.repeat(np.asarray(gids) - int(info['P'].gids[0]), counts))


def _brian2_setup(b2: Any) -> None:
    b2.start_scope()
    try:  # compiled code if a C compiler and Cython are present, else numpy
        import Cython  # noqa: F401
        b2.prefs.codegen.target = 'cython'
    except ImportError:  # pragma: no cover - environment dependent
        b2.prefs.codegen.target = 'numpy'
    b2.defaultclock.dt = COBA['dt'] * b2.ms


_EQS = '''
dv/dt = (gL * (EL - vb) + ge * (Eex - vb) + gi * (Ein - vb) + Ie) / Cm : volt (unless refractory)
vb = clip(v, -1 * volt, Vt) : volt
dge/dt = -ge / taue : siemens
dgi/dt = -gi / taui : siemens
'''


def _namespace(b2: Any, I_e: float) -> dict:
    c = COBA
    return dict(gL=c['g_L'] * b2.nS, EL=c['E_L'] * b2.mV, Eex=c['E_ex'] * b2.mV,
                Ein=c['E_in'] * b2.mV, Ie=I_e * b2.pA, Cm=c['C_m'] * b2.pF,
                Vt=c['V_th'] * b2.mV, Vr=c['V_reset'] * b2.mV, taue=c['tau_ex'] * b2.ms,
                taui=c['tau_in'] * b2.ms)


def _group(b2: Any, n: int, I_e: float = COBA['I_e']):
    c = COBA
    return b2.NeuronGroup(n, _EQS, threshold='v >= Vt', reset='v = Vr',
                          refractory=(c['t_ref'] + c['dt']) * b2.ms, method='rk4',
                          namespace=_namespace(b2, I_e))


def brian2_coba(seed: int, t_ms: float, scale: float = 1.0):
    """Spike events ``(times ms, senders)`` of one Brian2 COBA run (times shifted by ``+dt``)."""
    import brian2 as b2
    c = COBA
    _brian2_setup(b2)
    b2.seed(seed)
    ne, ni = int(round(c['N_E'] * scale)), int(round(c['N_I'] * scale))
    P = _group(b2, ne + ni)
    P.v = 'Vr + rand() * (Vt - Vr)'
    Ce = b2.Synapses(P[:ne], P, on_pre='ge += we', delay=c['dt'] * b2.ms,
                     namespace=dict(we=c['w_ex'] * b2.nS))
    Ci = b2.Synapses(P[ne:], P, on_pre='gi += wi', delay=c['dt'] * b2.ms,
                     namespace=dict(wi=c['w_in'] * b2.nS))
    Ce.connect(condition='i != j', p=c['p'])
    Ci.connect(condition=f'i + {ne} != j', p=c['p'])
    sm = b2.SpikeMonitor(P)
    net = b2.Network(P, Ce, Ci, sm)
    net.run(t_ms * b2.ms)
    return np.asarray(sm.t / b2.ms) + c['dt'], np.asarray(sm.i)


def kernel_single(ex_times: Any, in_times: Any, t_ms: float, I_e: float, substeps: int | None = None):
    """One kernel ``iaf_cond_exp`` (COBA parameters) driven by spike trains (float64).

    Returns
    -------
    dict
        ``V_m`` (mV), ``g_ex``, ``g_in`` (nS) traces ``(n_steps,)`` and ``spikes`` (ms).
    """
    import jax
    from brainpy_state._kernel.integrators import RK
    c = dict(COBA, I_e=I_e)
    with jax.enable_x64(True):
        net = Network(dt=COBA['dt'] * u.ms)
        kw = {} if substeps is None else dict(integrator=RK(4, substeps))
        n = net.add(iaf_cond_exp, 1, params=_neuron_params(c), name='n', **kw)
        d = COBA['dt'] * u.ms
        for times, port, w in ((ex_times, 'ex', c['w_ex']), (in_times, 'in_', c['w_in'])):
            sg = net.add(spike_generator, 1, params=dict(spike_times=np.asarray(times) * u.ms))
            net.connect(sg, n, port=port, weight=w * u.nS, delay=d)
        rec = net.observe(n, 'V_m', 'g_ex', 'g_in', 'spike')
        res = net.compile('float64').runner().simulate(t_ms * u.ms)
        r = res[rec]
        t, _, cnt = r.spike_events()
        return dict(V_m=np.asarray(r['V_m'].to_decimal(u.mV))[:, 0],
                    g_ex=np.asarray(r['g_ex'].to_decimal(u.nS))[:, 0],
                    g_in=np.asarray(r['g_in'].to_decimal(u.nS))[:, 0],
                    spikes=np.repeat(np.asarray(t.to_decimal(u.ms)), np.asarray(cnt, int)))


def brian2_single(ex_times: Any, in_times: Any, t_ms: float, I_e: float) -> dict:
    """The same neuron and inputs in Brian2 (``rk4``); traces recorded at the end of each step."""
    import brian2 as b2
    c = COBA
    _brian2_setup(b2)
    b2.prefs.codegen.target = 'numpy'  # one neuron: no compilation needed
    G = _group(b2, 1, I_e)
    G.v = c['E_L'] * b2.mV
    objs = [G]
    for times, var, w in ((ex_times, 'ge', c['w_ex']), (in_times, 'gi', c['w_in'])):
        times = np.asarray(times, float)
        sg = b2.SpikeGeneratorGroup(1, np.zeros(times.size, int), times * b2.ms)
        syn = b2.Synapses(sg, G, on_pre=f'{var} += w', namespace=dict(w=w * b2.nS))
        syn.connect()
        objs += [sg, syn]
    M = b2.StateMonitor(G, ['v', 'ge', 'gi'], record=0, when='end')
    sm = b2.SpikeMonitor(G)
    net = b2.Network(*objs, M, sm)
    net.run(t_ms * b2.ms)
    return dict(V_m=np.asarray(M.v[0] / b2.mV), g_ex=np.asarray(M.ge[0] / b2.nS),
                g_in=np.asarray(M.gi[0] / b2.nS), spikes=np.asarray(sm.t / b2.ms) + c['dt'])
