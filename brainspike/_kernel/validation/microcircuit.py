# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Potjans--Diesmann (2014) cortical microcircuit on the kernel and in NEST (gate K1a-G4).

Parameters follow the reference PyNEST implementation
(``INM-6/microcircuit-PD14-model``, ``PyNEST/src/microcircuit/network_params.py``):
eight populations (L2/3, L4, L5, L6; E and I), 77 169 neurons at full scale,
``fixed_total_number`` connectivity with
:math:`N_{syn} = \ln(1 - p) / \ln((N_{pre} N_{post} - 1) / (N_{pre} N_{post}))`
(autapses and multapses allowed), weights and delays from normal distributions
redrawn into ``[0, inf)`` / ``(-inf, 0]`` (weights) and ``[dt / 2, inf)`` (delays),
``iaf_psc_exp`` neurons (``tau_syn = 0.5 ms``, ``tau_m = 10 ms``, ``C_m = 250 pF``,
``t_ref = 2 ms``), "optimized" population-specific initial ``V_m``, and Poisson
background input ``bg_rate * K_ext`` per neuron (``bg_input='poisson'``, the original
model) or the compensating DC current (``bg_input='dc'``, the reference repository's
current default). Thalamic and transient DC stimulation are off (their defaults).

The neurons are the model library's
:class:`~brainspike._kernel.models.iaf_psc_exp` (promoted from this module in
K1a-7; NEST parity in ``models/nest_parity_test.py``). :class:`TruncatedNormal` gives NEST's
``nest.math.redraw(nest.random.normal(...))`` distribution (redraw, not clip).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import brainunit as u
import jax
import numpy as np

from brainspike._kernel.models import iaf_psc_exp
from brainspike._kernel.params import Distribution
from brainspike._kernel.units import check_unit, to_mantissa, unit_of

__all__ = ['TruncatedNormal', 'POPULATIONS', 'pd14_params',
           'build_microcircuit', 'nest_microcircuit', 'kernel_microcircuit', 'population_rates']

POPULATIONS = ('L23E', 'L23I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I')
FULL_NUM_NEURONS = np.array([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948])
CONN_PROBS = np.array([
    [0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0, 0.0076, 0.0],
    [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0, 0.0042, 0.0],
    [0.0077, 0.0059, 0.0497, 0.135, 0.0067, 0.0003, 0.0453, 0.0],
    [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0, 0.1057, 0.0],
    [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0],
    [0.0548, 0.0269, 0.0257, 0.0022, 0.06, 0.3158, 0.0086, 0.0],
    [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
    [0.0364, 0.001, 0.0034, 0.0005, 0.0277, 0.008, 0.0658, 0.1443],
])  # [target, source]
K_EXT = np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100])
V0_MEAN = np.array([-68.28, -63.16, -63.33, -63.45, -63.11, -61.66, -66.72, -61.43])
V0_STD = np.array([5.36, 4.57, 4.74, 4.94, 4.94, 4.55, 5.46, 4.48])
NEURON = dict(E_L=-65.0, V_th=-50.0, V_reset=-65.0, C_m=250.0, tau_m=10.0, tau_syn=0.5, t_ref=2.0)
PSP_EXC, G, W_REL_STD = 0.15, -4.0, 0.1
DELAY_EXC, DELAY_INH, DELAY_REL_STD = 1.5, 0.75, 0.5
BG_RATE, DELAY_POISSON = 8.0, 1.5

_PA = u.pA


# ---------------------------------------------------------------------------
# distribution
# ---------------------------------------------------------------------------

@dataclass(frozen=True, init=False)
class TruncatedNormal(Distribution):
    """Normal distribution redrawn into ``[low, high]`` (NEST ``nest.math.redraw``).

    Unlike ``Normal(..., low=, high=)`` (which clips, putting mass on the bounds),
    the samples follow the normal density restricted to the interval.

    Parameters
    ----------
    mean, std : float or brainunit.Quantity
        Mean and standard deviation (``std > 0``) of the untruncated normal.
    low, high : float or brainunit.Quantity, optional
        Bounds (default unbounded on that side).

    Examples
    --------
    .. code-block:: python

        >>> import jax
        >>> import brainunit as u
        >>> from brainspike._kernel.validation.microcircuit import TruncatedNormal
        >>> d = TruncatedNormal(1.5 * u.ms, 0.75 * u.ms, low=0.05 * u.ms)
        >>> x = d.sample(jax.random.PRNGKey(0), (10000,)).to_decimal(u.ms)
        >>> bool(x.min() >= 0.05)
        True
    """

    mean: float
    std: float
    unit: u.Unit
    low: float | None
    high: float | None

    def __init__(self, mean: Any, std: Any, *, low: Any = None, high: Any = None) -> None:
        unit = unit_of(mean)
        check_unit(std, unit, 'std')
        object.__setattr__(self, 'unit', unit)
        object.__setattr__(self, 'mean', float(np.asarray(to_mantissa(mean, unit))))
        object.__setattr__(self, 'std', float(np.asarray(to_mantissa(std, unit))))
        if not self.std > 0:
            raise ValueError(f'std must be positive, got {std}')
        self._init_clip(low, high)

    def _sample_mantissa(self, key, shape, dtype):
        lo = -np.inf if self.low is None else (self.low - self.mean) / self.std
        hi = np.inf if self.high is None else (self.high - self.mean) / self.std
        z = jax.random.truncated_normal(key, lo, hi, shape, dtype)
        return self.mean + self.std * z

    def _support(self):
        return -np.inf, np.inf


# ---------------------------------------------------------------------------
# parameters and builders
# ---------------------------------------------------------------------------

def _psc_over_psp(c_m: float, tau_m: float, tau_syn: float) -> float:
    sub = 1.0 / (tau_syn - tau_m)
    pre = tau_m * tau_syn / c_m * sub
    frac = (tau_m / tau_syn) ** sub
    return 1.0 / (pre * (frac ** tau_m - frac ** tau_syn))


def pd14_params(scale: float = 1.0) -> dict:
    """Derived parameters (reference ``network.py::__derive_parameters``, ``K_scaling = 1``).

    Parameters
    ----------
    scale : float, default 1.0
        ``N_scaling`` (neurons and synapse counts scale together, in-degrees are
        kept). The gate runs ``1.0``; smaller values are for tests only.

    Returns
    -------
    dict
        ``n`` (8,), ``n_syn`` (8, 8) ``[target, source]``, ``w_mean`` (8, 8) pA,
        ``d_mean`` (8, 8) ms, ``w_ext`` pA, ``k_ext`` (8,), ``dc`` (8,) pA (the
        compensating current of ``bg_input='dc'``).

    Examples
    --------
    .. code-block:: python

        >>> from brainspike._kernel.validation.microcircuit import pd14_params
        >>> p = pd14_params()
        >>> int(p['n'].sum()), round(float(p['n_syn'].sum()) / 1e8, 2)
        (77169, 2.99)
    """
    n_full = FULL_NUM_NEURONS
    prod = np.outer(n_full, n_full).astype(float)
    full_syn = np.log(1.0 - CONN_PROBS) / np.log((prod - 1.0) / prod)
    n = np.round(n_full * scale).astype(int)
    n_syn = np.round(full_syn * scale).astype(int)
    psc = _psc_over_psp(NEURON['C_m'], NEURON['tau_m'], NEURON['tau_syn'])
    psp = np.zeros((8, 8))
    psp[:, 0::2] = PSP_EXC
    psp[:, 1::2] = PSP_EXC * G
    psp[0, 2] = 2.0 * PSP_EXC
    d = np.zeros((8, 8))
    d[:, 0::2] = DELAY_EXC
    d[:, 1::2] = DELAY_INH
    w_ext = PSP_EXC * psc
    return dict(n=n, n_syn=n_syn, w_mean=psp * psc, d_mean=d, w_ext=w_ext, k_ext=K_EXT.copy(),
                dc=BG_RATE * K_EXT * w_ext * NEURON['tau_syn'] * 1e-3, psc_over_psp=psc)


def build_microcircuit(scale: float = 1.0, *, seed: int = 0, dt: float = 0.1,
                       bg_input: str = 'poisson', record_interval: float | None = 100.0):
    """Build the microcircuit as a kernel :class:`~brainspike._kernel.network.Network`.

    Parameters
    ----------
    scale : float, default 1.0
        Neuron-number scaling (``1.0`` = 77 169 neurons, ~3e8 synapses).
    seed : int, default 0
        Network seed (connectivity, weights, delays, initial ``V_m``, Poisson input).
    dt : float, default 0.1
        Step in ms.
    bg_input : {'poisson', 'dc'}, default 'poisson'
        Background input (see the module docstring).
    record_interval : float or None, default 100.0
        If given, every population's spike counts are recorded, summed over
        windows of this length (ms) -- enough for population rates.

    Returns
    -------
    net : Network
    info : dict
        ``pops`` (8 views), ``obs`` (8 observers or ``None``), ``params``.
    """
    from brainspike._kernel.connectivity import fixed_total_number, one_to_one
    from brainspike._kernel.models import poisson_generator
    from brainspike._kernel.network import Network

    if bg_input not in ('poisson', 'dc'):
        raise ValueError(f"bg_input must be 'poisson' or 'dc', got {bg_input!r}")
    p = pd14_params(scale)
    net = Network(dt=dt * u.ms, seed=seed)
    pops = []
    for i, name in enumerate(POPULATIONS):
        npar = dict(C_m=NEURON['C_m'] * u.pF, tau_m=NEURON['tau_m'] * u.ms,
                    tau_syn_ex=NEURON['tau_syn'] * u.ms, tau_syn_in=NEURON['tau_syn'] * u.ms,
                    t_ref=NEURON['t_ref'] * u.ms, E_L=NEURON['E_L'] * u.mV,
                    V_th=NEURON['V_th'] * u.mV, V_reset=NEURON['V_reset'] * u.mV,
                    I_e=(p['dc'][i] if bg_input == 'dc' else 0.0) * u.pA)
        init = dict(V_m=TruncatedNormal(V0_MEAN[i] * u.mV, V0_STD[i] * u.mV))
        pops.append(net.add(iaf_psc_exp, int(p['n'][i]), params=npar, init=init, name=name))
    if bg_input == 'poisson':
        for i, name in enumerate(POPULATIONS):
            bg = net.add(poisson_generator, int(p['n'][i]),
                         params=dict(rate=BG_RATE * p['k_ext'][i] * u.Hz), name=f'bg_{name}')
            net.connect(bg, pops[i], one_to_one(), port='ex', weight=p['w_ext'] * u.pA,
                        delay=DELAY_POISSON * u.ms, name=f'bg_to_{name}')
    for i in range(8):        # target
        for j in range(8):    # source
            n_syn = int(p['n_syn'][i, j])
            if n_syn <= 0:
                continue
            w, d = float(p['w_mean'][i, j]), float(p['d_mean'][i, j])
            if w >= 0:
                weight = TruncatedNormal(w * u.pA, abs(w) * W_REL_STD * u.pA, low=0.0 * u.pA)
                port = 'ex'
            else:
                weight = TruncatedNormal(w * u.pA, abs(w) * W_REL_STD * u.pA, high=0.0 * u.pA)
                port = 'in_'
            delay = TruncatedNormal(d * u.ms, d * DELAY_REL_STD * u.ms, low=0.5 * dt * u.ms)
            net.connect(pops[j], pops[i], fixed_total_number(n_syn), port=port, weight=weight,
                        delay=delay, name=f'{POPULATIONS[j]}_{POPULATIONS[i]}')
    obs = None
    if record_interval is not None:
        obs = [net.observe(pp, 'spike', interval=record_interval * u.ms, name=f'rec_{nm}')
               for pp, nm in zip(pops, POPULATIONS)]
    return net, dict(pops=pops, obs=obs, params=p)


def population_rates(counts: Mapping[str, Any], n: Any, duration_ms: float) -> dict:
    """``{population: rate in Hz}`` from total spike counts per population."""
    return {k: float(counts[k]) / (float(n[i]) * duration_ms * 1e-3)
            for i, k in enumerate(POPULATIONS)}


def nest_microcircuit(seed: int, t_warm: float, t_stat: float, *, scale: float = 1.0,
                      threads: int = 16, bg_input: str = 'poisson', dt: float = 0.1) -> dict:
    """One NEST 3.10 run of the same network.

    Returns
    -------
    dict
        ``rates`` (``{population: Hz}`` over ``[t_warm, t_warm + t_stat)``),
        ``build_s``, ``sim_s``, ``n_syn`` (realised synapse count).
    """
    import time

    import nest
    p = pd14_params(scale)
    nest.ResetKernel()
    nest.set_verbosity('M_ERROR')
    nest.resolution = dt
    nest.rng_seed = seed
    nest.local_num_threads = threads
    t0 = time.perf_counter()
    pops = []
    for i in range(8):
        pop = nest.Create('iaf_psc_exp', int(p['n'][i]))
        pop.set(tau_syn_ex=NEURON['tau_syn'], tau_syn_in=NEURON['tau_syn'], E_L=NEURON['E_L'],
                V_th=NEURON['V_th'], V_reset=NEURON['V_reset'], t_ref=NEURON['t_ref'],
                C_m=NEURON['C_m'], tau_m=NEURON['tau_m'],
                I_e=float(p['dc'][i]) if bg_input == 'dc' else 0.0)
        pop.set(V_m=nest.random.normal(float(V0_MEAN[i]), float(V0_STD[i])))
        pops.append(pop)
    recs = nest.Create('spike_recorder', 8, params={'start': t_warm, 'stop': t_warm + t_stat})
    if bg_input == 'poisson':
        bg = nest.Create('poisson_generator', 8)
        bg.rate = (BG_RATE * p['k_ext']).astype(float).tolist()
    for i in range(8):
        for j in range(8):
            n_syn = int(p['n_syn'][i, j])
            if n_syn <= 0:
                continue
            w, d = float(p['w_mean'][i, j]), float(p['d_mean'][i, j])
            w_min, w_max = (0.0, np.inf) if w >= 0 else (-np.inf, 0.0)
            syn = {'synapse_model': 'static_synapse',
                   'weight': nest.math.redraw(nest.random.normal(mean=w, std=abs(w * W_REL_STD)),
                                              min=w_min, max=w_max),
                   'delay': nest.math.redraw(nest.random.normal(mean=d, std=d * DELAY_REL_STD),
                                             min=nest.resolution - 0.5 * nest.resolution,
                                             max=np.inf)}
            nest.Connect(pops[j], pops[i], {'rule': 'fixed_total_number', 'N': n_syn}, syn)
    for i in range(8):
        nest.Connect(pops[i], recs[i])
        if bg_input == 'poisson':
            nest.Connect(bg[i], pops[i], 'all_to_all',
                         {'weight': float(p['w_ext']), 'delay': DELAY_POISSON})
    build_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    nest.Simulate(t_warm + t_stat)
    sim_s = time.perf_counter() - t1
    counts = {k: int(recs[i].n_events) for i, k in enumerate(POPULATIONS)}
    return dict(rates=population_rates(counts, p['n'], t_stat), build_s=build_s, sim_s=sim_s,
                n_syn=int(nest.num_connections))


def kernel_microcircuit(seed: int, t_warm: float, t_stat: float, *, scale: float = 1.0,
                        precision: str = 'float32', bg_input: str = 'poisson', dt: float = 0.1,
                        chunk_ms: float = 100.0) -> dict:
    """One kernel run, measured: build, compile, step time and population rates.

    The run is one ``cont(t_warm)`` call (trace + compile + run), then
    ``t_stat / chunk_ms`` timed ``cont(chunk_ms)`` calls of the same compiled rollout;
    spike counts are recorded per ``chunk_ms`` window. ``t_warm`` must equal
    ``chunk_ms``: the compile time is the first call minus the median warm call of
    the same length.

    Returns
    -------
    dict
        ``rates`` (``{population: Hz}`` over the ``t_stat`` window), ``build_s``
        (network, compile to a ``Plan``, runner state), ``first_s``, ``compile_s``,
        ``chunk_s`` (list), ``ms_per_step`` (median warm chunk), ``n_edges``
        (recurrent + background edges).
    """
    import time

    n_chunks = t_stat / chunk_ms
    if abs(t_warm - chunk_ms) > 1e-9 or abs(n_chunks - round(n_chunks)) > 1e-9 or n_chunks < 1:
        raise ValueError('t_warm must equal chunk_ms and t_stat be a positive multiple of it')
    t0 = time.perf_counter()
    net, info = build_microcircuit(scale, seed=seed, dt=dt, bg_input=bg_input,
                                   record_interval=chunk_ms)
    plan = net.compile(precision)
    run = plan.runner()
    jax.block_until_ready(run.state)
    build_s = time.perf_counter() - t0
    obs = info['obs']

    def chunk():
        t = time.perf_counter()
        res = run.cont(chunk_ms * u.ms)
        c = {k: float(np.asarray(res[o]['spike']).sum()) for k, o in zip(POPULATIONS, obs)}
        return time.perf_counter() - t, c

    first_s, _ = chunk()
    counts = dict.fromkeys(POPULATIONS, 0.0)
    times = []
    for _ in range(int(round(n_chunks))):
        s, c = chunk()
        times.append(s)
        for k in POPULATIONS:
            counts[k] += c[k]
    warm = float(np.median(times))
    n_edges = int(sum(pp.edges.n_edges for pp in plan.projections))
    return dict(rates=population_rates(counts, info['params']['n'], t_stat), build_s=build_s,
                first_s=first_s, compile_s=first_s - warm, chunk_s=times,
                ms_per_step=1e3 * warm / int(round(chunk_ms / dt)), n_edges=n_edges)
