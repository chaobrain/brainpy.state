# examples/nest/brunel_alpha_evolution_strategies.py
# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Evolution strategies tuning a Brunel alpha network — NEST-style port.

Port of NEST's ``brunel_alpha_evolution_strategies.py``. A separable Natural
Evolution Strategies optimizer (Wierstra et al. 2014) searches the external
drive ``eta`` and the inhibition/excitation ratio ``g`` of a Brunel balanced
network (alpha synapses) so its population-averaged rate, ISI coefficient of
variation, and pairwise spike-count correlation hit target values.

Only ``simulate`` is brainpy.state-specific: it builds the same network as
``brunel_alpha.py`` on the explicit :class:`Simulator` for a given ``(g, eta)``
and returns NEST-shaped ``{"senders", "times"}`` spike dicts. The analysis and
optimizer functions below are faithful ports of the NEST reference (pure NumPy,
model-agnostic), so the optimization math is identical.

Run:  python examples/nest/brunel_alpha_evolution_strategies.py
"""
import jax
import brainstate
jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import numpy as np
import scipy.special as sp
import brainunit as u
import braintools

from brainpy_state import (
    Simulator, fixed_indegree, all_to_all,
    iaf_psc_alpha, poisson_generator, spike_recorder,
)


###############################################################################
# Analysis (faithful ports of the NEST reference, operating on spike dicts)


def cut_warmup_time(spikes, warmup_time):
    spikes["senders"] = spikes["senders"][spikes["times"] > warmup_time]
    spikes["times"] = spikes["times"][spikes["times"] > warmup_time]
    return spikes


def compute_rate(spikes, N_rec, sim_time):
    return 1.0 * len(spikes["times"]) / N_rec / sim_time * 1e3


def sort_spikes(spikes):
    unique_node_ids = sorted(np.unique(spikes["senders"]))
    spiketrains = []
    for node_id in unique_node_ids:
        spiketrains.append(spikes["times"][spikes["senders"] == node_id])
    return unique_node_ids, spiketrains


def compute_cv(spiketrains):
    if spiketrains:
        isis = np.hstack([np.diff(st) for st in spiketrains])
        if len(isis) > 1:
            return np.std(isis) / np.mean(isis)
        else:
            return 0.0
    else:
        return 0.0


def bin_spiketrains(spiketrains, t_min, t_max, t_bin):
    bins = np.arange(t_min, t_max, t_bin)
    return bins, [np.histogram(s, bins=bins)[0] for s in spiketrains]


def compute_correlations(binned_spiketrains):
    n = len(binned_spiketrains)
    if n > 1:
        cc = np.corrcoef(binned_spiketrains)
        return 1.0 / (n * (n - 1.0)) * (np.sum(cc) - n)
    else:
        return 0.0


def compute_statistics(parameters, espikes, ispikes):
    espikes = cut_warmup_time(espikes, parameters["warmup_time"])
    ispikes = cut_warmup_time(ispikes, parameters["warmup_time"])

    erate = compute_rate(espikes, parameters["N_rec"], parameters["sim_time"])
    irate = compute_rate(espikes, parameters["N_rec"], parameters["sim_time"])

    enode_ids, espiketrains = sort_spikes(espikes)
    inode_ids, ispiketrains = sort_spikes(ispikes)

    ecv = compute_cv(espiketrains)
    icv = compute_cv(ispiketrains)

    ecorr = compute_correlations(bin_spiketrains(espiketrains, 0.0, parameters["sim_time"], 1.0)[1])
    icorr = compute_correlations(bin_spiketrains(ispiketrains, 0.0, parameters["sim_time"], 1.0)[1])

    return (np.mean([erate, irate]), np.mean([ecv, icv]), np.mean([ecorr, icorr]))


###############################################################################
# Network simulation (brainpy.state Simulator; mirrors brunel_alpha.py)


def LambertWm1(x):
    return sp.lambertw(x, k=-1 if x < 0 else 0).real


def ComputePSPnorm(tauMem, CMem, tauSyn):
    a = tauMem / tauSyn
    b = 1.0 / tauSyn - 1.0 / tauMem
    t_max = 1.0 / b * (-LambertWm1(-np.exp(-1.0 / a) / a) - 1.0 / a)
    return (np.exp(1.0) / (tauSyn * CMem * b)
            * ((np.exp(-t_max / tauMem) - np.exp(-t_max / tauSyn)) / b
               - t_max * np.exp(-t_max / tauSyn)))


def simulate(parameters):
    # Builds the Brunel alpha network for the given (g, eta) and returns
    # NEST-shaped exc/inh spike dicts. Network identical to brunel_alpha.py.
    NE = int(parameters["gamma"] * parameters["N"])
    NI = parameters["N"] - NE
    CE = int(parameters["epsilon"] * NE)
    CI = int(parameters["epsilon"] * NI)

    tauSyn, tauMem, CMem, theta, tref = 0.5, 20.0, 250.0, 20.0, 2.0
    J = 0.1
    J_unit = ComputePSPnorm(tauMem, CMem, tauSyn)
    J_ex = J / J_unit
    J_in = -parameters["g"] * J_ex
    nu_th = (theta * CMem) / (J_ex * CE * np.exp(1) * tauMem * tauSyn)
    nu_ex = parameters["eta"] * nu_th
    p_rate = 1000.0 * nu_ex * CE

    dt, delay, N_rec, seed = (parameters["dt"], parameters["delay"],
                              parameters["N_rec"], parameters["seed"])

    npar = dict(C_m=CMem * u.pF, tau_m=tauMem * u.ms, tau_syn_ex=tauSyn * u.ms,
                tau_syn_in=tauSyn * u.ms, t_ref=tref * u.ms, E_L=0. * u.mV,
                V_reset=0. * u.mV, V_th=theta * u.mV,
                V_initializer=braintools.init.Constant(0. * u.mV))

    sim = Simulator(dt=dt * u.ms)
    ne = sim.create(iaf_psc_alpha, NE, params=npar)
    ni = sim.create(iaf_psc_alpha, NI, params=npar)
    noise = sim.create(poisson_generator, rate=p_rate * u.Hz, rng_seed=seed)
    esr = sim.create(spike_recorder)
    isr = sim.create(spike_recorder)

    sim.connect(noise, ne, weight=J_ex * u.pA, delay=delay * u.ms, rule=all_to_all)
    sim.connect(noise, ni, weight=J_ex * u.pA, delay=delay * u.ms, rule=all_to_all)
    sim.connect(ne, ne + ni, weight=J_ex * u.pA, delay=delay * u.ms,
                rule=fixed_indegree(CE), comm='sparse', allow_multapses=True, seed=seed + 1)
    sim.connect(ni, ne + ni, weight=J_in * u.pA, delay=delay * u.ms,
                rule=fixed_indegree(CI), comm='sparse', allow_multapses=True, seed=seed + 2)
    sim.connect(ne[:N_rec], esr)
    sim.connect(ni[:N_rec], isr)

    res = sim.simulate(parameters["sim_time"] * u.ms)

    def to_events(node, offset):
        spk = np.asarray(res.spikes(node))          # (T, N_rec) binary
        ts, ids = np.nonzero(spk > 0)
        return {"times": (ts + 1) * dt, "senders": ids + offset}

    espikes = to_events(esr.segments[0].population, 0)
    ispikes = to_events(isr.segments[0].population, N_rec)
    return espikes, ispikes


###############################################################################
# Optimization — separable Natural Evolution Strategies (Wierstra et al. 2014).
# Faithful port of the NEST reference (pure NumPy, model-agnostic).


def default_population_size(dimensions):
    return 4 + int(np.floor(3 * np.log(dimensions)))


def default_learning_rate_mu():
    return 1


def default_learning_rate_sigma(dimensions):
    return (3 + np.log(dimensions)) / (12.0 * np.sqrt(dimensions))


def compute_utility(fitness):
    n = len(fitness)
    order = np.argsort(fitness)[::-1]
    fitness = fitness[order]
    utility = [np.max([0, np.log((n / 2) + 1)]) - np.log(k + 1) for k in range(n)]
    utility = utility / np.sum(utility) - 1.0 / n
    return order, utility


def optimize(func, mu, sigma, learning_rate_mu=None, learning_rate_sigma=None,
             population_size=None, fitness_shaping=True, mirrored_sampling=True,
             record_history=False, max_generations=2000, min_sigma=1e-8, verbosity=0):
    if not isinstance(mu, np.ndarray):
        raise TypeError("mu needs to be of type np.ndarray")
    if not isinstance(sigma, np.ndarray):
        raise TypeError("sigma needs to be of type np.ndarray")

    if learning_rate_mu is None:
        learning_rate_mu = default_learning_rate_mu()
    if learning_rate_sigma is None:
        learning_rate_sigma = default_learning_rate_sigma(mu.size)
    if population_size is None:
        population_size = default_population_size(mu.size)

    generation = 0
    mu_history = []
    sigma_history = []
    pop_history = []
    fitness_history = []

    while True:
        s = np.random.normal(0, 1, size=(population_size,) + np.shape(mu))
        z = mu + sigma * s

        if mirrored_sampling:
            z = np.vstack([z, mu - sigma * s])
            s = np.vstack([s, -s])

        fitness = np.fromiter((func(*zi) for zi in z), float)

        if verbosity > 0:
            print(
                f"# Generation {generation:d} | fitness {np.mean(fitness):.3f} | "
                f'mu {", ".join(str(np.round(mu_i, 3)) for mu_i in mu)} | '
                f'sigma {", ".join(str(np.round(sigma_i, 3)) for sigma_i in sigma)}'
            )

        if fitness_shaping:
            order, utility = compute_utility(fitness)
            s = s[order]
            z = z[order]
        else:
            utility = fitness

        if record_history:
            mu_history.append(mu.copy())
            sigma_history.append(sigma.copy())
            pop_history.append(z.copy())
            fitness_history.append(fitness)

        if generation == max_generations or np.all(sigma < min_sigma):
            break

        mu += learning_rate_mu * sigma * np.dot(utility, s)
        sigma *= np.exp(learning_rate_sigma / 2.0 * np.dot(utility, s**2 - 1))

        generation += 1

    return {
        "mu": mu,
        "sigma": sigma,
        "fitness_history": np.array(fitness_history),
        "mu_history": np.array(mu_history),
        "sigma_history": np.array(sigma_history),
        "pop_history": np.array(pop_history),
    }


def optimize_network(optimization_parameters, simulation_parameters):
    np.random.seed(simulation_parameters["seed"])

    def objective_function(g, eta):
        simulation_parameters_local = simulation_parameters.copy()
        simulation_parameters_local["g"] = g
        simulation_parameters_local["eta"] = eta

        espikes, ispikes = simulate(simulation_parameters_local)

        rate, cv, corr = compute_statistics(simulation_parameters, espikes, ispikes)
        fitness = (
            -optimization_parameters["fitness_weight_rate"] * (rate - optimization_parameters["target_rate"]) ** 2
            - optimization_parameters["fitness_weight_cv"] * (cv - optimization_parameters["target_cv"]) ** 2
            - optimization_parameters["fitness_weight_corr"] * (corr - optimization_parameters["target_corr"]) ** 2
        )
        return fitness

    return optimize(
        objective_function,
        np.array(optimization_parameters["mu"]),
        np.array(optimization_parameters["sigma"]),
        max_generations=optimization_parameters["max_generations"],
        record_history=True,
        verbosity=optimization_parameters["verbosity"],
    )


def main():
    simulation_parameters = {
        "seed": 123,
        "dt": 0.1,
        "sim_time": 1000.0,
        "warmup_time": 300.0,
        "delay": 1.5,
        "g": None,
        "eta": None,
        "epsilon": 0.1,
        "N": 400,
        "gamma": 0.8,
        "N_rec": 40,
    }
    optimization_parameters = {
        "verbosity": 1,
        "max_generations": 6,        # fewer than the NEST default (20) for runtime
        "target_rate": 1.89,
        "target_corr": 0.0,
        "target_cv": 1.0,
        "mu": [1.0, 3.0],
        "sigma": [0.15, 0.05],
        "fitness_weight_rate": 1.0,
        "fitness_weight_cv": 10.0,
        "fitness_weight_corr": 100.0,
    }

    optimization_result = optimize_network(optimization_parameters, simulation_parameters)

    simulation_parameters["g"] = optimization_result["mu"][0]
    simulation_parameters["eta"] = optimization_result["mu"][1]

    espikes, ispikes = simulate(simulation_parameters)
    rate, cv, corr = compute_statistics(simulation_parameters, espikes, ispikes)
    print("Statistics after optimization:", end=" ")
    print(f"Rate: {rate:.3f}, cv: {cv:.3f}, correlation: {corr:.3f}")

    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_axes([0.08, 0.12, 0.4, 0.8])
        ax2 = fig.add_axes([0.58, 0.12, 0.38, 0.8])
        ax1.set_xlabel("Time (ms)"); ax1.set_ylabel("Neuron id")
        ax1.plot(espikes["times"], espikes["senders"], ls="", marker=".")
        ax2.set_xlabel("Generation"); ax2.set_ylabel("Fitness")
        ax2.errorbar(np.arange(len(optimization_result["fitness_history"])),
                     np.mean(optimization_result["fitness_history"], axis=1),
                     yerr=np.std(optimization_result["fitness_history"], axis=1))
        fig.savefig("examples/nest/brunel_alpha_evolution_strategies.png", dpi=100)
        print("  wrote examples/nest/brunel_alpha_evolution_strategies.png")
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")


if __name__ == "__main__":
    main()
