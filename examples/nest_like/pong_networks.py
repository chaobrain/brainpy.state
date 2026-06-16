# examples/nest_like/pong_networks.py
r"""Spiking networks that learn to play Pong — NEST §3.10 ``pong/networks.py`` port.

Two learners encode an input→output mapping in the weights between an input layer
(one neuron per ball y-cell) and a motor layer (one neuron per paddle target cell):

* :class:`PongNetRSTDP` — static synapses whose weights are rewritten on the host
  after every 200 ms turn by the reward-modulated STDP rule of Wunderlich et al.
  (2019): a scalar reward (closeness of the winning motor neuron to the target
  cell) times a per-edge spike-timing correlation.
* :class:`PongNetDopa` *(added in Stage E)* — dopaminergic ``stdp_dopamine_synapse``
  edges driven by an actor–critic circuit; the reward is injected as a host-set
  current into the critic's dopaminergic neurons.

Both share :class:`PongNetBase`, which builds the input/motor layers and the
turn-by-turn bookkeeping (reward baseline, winning neuron, performance history).

The host drives the network in 200 ms turns with :meth:`Simulator.cont` (state
persists across turns — biological time accumulates). The active input cell is set
between turns through a :class:`host_spike_drive` schedule, and R-STDP overwrites the
static input→motor weights in place with :meth:`SynapseCollection.set` — neither
recompiles the per-turn rollout (every per-turn change is a fixed-shape State write).

Reference
---------
Wunderlich T. et al. (2019), *Demonstrating advantages of neuromorphic computation:
a pilot study.* Front. Neurosci. 13:260. Original implementation:
https://github.com/electronicvisions/model-sw-pong

Run (host loop + comparison harness): ``examples/nest_like/pong_run.py``.
"""
import jax
import brainstate
import numpy as np
import brainunit as u

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64)

from brainpy.state import (Simulator, host_spike_drive, host_current_drive,
                           parrot_neuron, iaf_psc_exp, spike_recorder, noise_generator,
                           poisson_generator, static_synapse, stdp_dopamine_synapse,
                           volume_transmitter, one_to_one, all_to_all)

#: Integration step (ms); all §3.10 demos run at 0.1 ms / float64 / CPU.
DT = 0.1
#: Simulation time per turn (ms).
POLL_TIME = 200
#: Number of spikes in an input spiketrain per turn.
N_INPUT_SPIKES = 20
#: Inter-spike interval of the input spiketrain (ms).
ISI = 10.0
#: Standard deviation of the Gaussian background current (pA) for the noisy variant.
BG_STD = 220.0
#: Reward as a function of distance between winning and target neuron.
REWARDS_DICT = {0: 1.0, 1: 0.7, 2: 0.4, 3: 0.1}


def _to_pA(weight_q):
    """Bare-pA mantissa of a (live) weight Quantity from a ``SynapseCollection``."""
    return np.asarray(u.Quantity(weight_q).to_decimal(u.pA))


def calculate_stdp(pre_spikes, post_spikes, *, stdp_amplitude=36.0, stdp_tau=64.0,
                   stdp_saturation=128, only_causal=True, next_neighbor=True):
    r"""Accumulated R-STDP correlation between a pre- and a post-synaptic train.

    A faithful translation of NEST ``PongNetRSTDP.calculate_stdp`` — pure float math
    on spike *times* (ms), independent of any spiking machinery. For each post spike,
    the nearest preceding pre spike contributes facilitation
    ``A exp(-Δt/τ)`` and the nearest following pre spike contributes depression; with
    ``next_neighbor`` only one post spike per pre-interval is counted, and with
    ``only_causal`` only facilitation is returned. The result is clipped to
    ``stdp_saturation``.

    The computation is translation-invariant (it depends only on pre/post time
    *differences*), so chunk-local spike times give the same value as NEST's absolute
    biological times.

    Parameters
    ----------
    pre_spikes, post_spikes : array_like
        Pre- / post-synaptic spike times in ms (unsorted ok; sorted internally).
    stdp_amplitude : float, optional
        STDP curve amplitude ``A`` (arbitrary units). Default ``36.0``.
    stdp_tau : float, optional
        STDP time constant ``τ`` (ms). Default ``64.0``.
    stdp_saturation : float, optional
        Saturation clip for the accumulated trace. Default ``128``.
    only_causal : bool, optional
        Return only facilitation (no depression). Default ``True``.
    next_neighbor : bool, optional
        Count only next-neighbour coincidences (one post per pre-interval).
        Default ``True``.

    Returns
    -------
    float
        Accumulated (clipped) STDP correlation.
    """
    pre_spikes = np.sort(np.asarray(pre_spikes, dtype=float))
    post_spikes = np.sort(np.asarray(post_spikes, dtype=float))
    facilitation = 0.0
    depression = 0.0
    positions = np.searchsorted(pre_spikes, post_spikes)
    last_position = -1
    for spike, position in zip(post_spikes, positions):
        if position == last_position and next_neighbor:
            continue  # only next-neighbor pairs
        if position > 0:
            before_spike = pre_spikes[position - 1]
            facilitation += stdp_amplitude * np.exp(-(spike - before_spike) / stdp_tau)
        if position < len(pre_spikes):
            after_spike = pre_spikes[position]
            depression += stdp_amplitude * np.exp(-(after_spike - spike) / stdp_tau)
        last_position = position
    if only_causal:
        return min(facilitation, stdp_saturation)
    return min(facilitation - depression, stdp_saturation)


class PongNetBase:
    """Shared input/motor layers and turn bookkeeping for the two pong learners.

    Builds, on its own :class:`Simulator`, an input layer (a
    :class:`host_spike_drive` clamped through one ``parrot`` per cell) and a motor
    layer (one ``iaf_psc_exp`` per cell, tapped by one ``spike_recorder``). The
    input→motor connection is left to the subclass (it depends on the plasticity
    rule). The host drives the network one 200 ms turn at a time with
    :meth:`Simulator.cont`.

    Parameters
    ----------
    apply_noise : bool, optional
        If ``True`` the subclass adds background noise to the motor neurons (and
        uses the matching weight regime). Default ``True``.
    num_neurons : int, optional
        Neurons in both the input and motor layer (must match the game's ``y_grid``).
        Default ``20``.
    seed : int, optional
        Seed for the network's private RNG (initial weights + winning-neuron
        tie-break), kept separate from the game's global ``numpy`` RNG. Default ``0``.
    """

    #: Offset (ms) of the input spiketrain within each turn; set by the subclass.
    input_t_offset = 0

    def __init__(self, apply_noise=True, num_neurons=20, *, seed=0):
        self.apply_noise = apply_noise
        self.num_neurons = num_neurons
        self.rng = np.random.default_rng(seed)

        self.weight_history = []
        self.mean_reward = np.zeros(num_neurons)
        self.mean_reward_history = []
        self.winning_neuron = 0
        self.target_index = 0
        self.input_train = np.zeros(0)

        self.sim = Simulator(dt=DT * u.ms)
        self.poll_steps = int(round(POLL_TIME / DT))

        # Input: host-clamped per-step drive -> one parrot per cell (relays the train).
        drive_view = self.sim.create(host_spike_drive, num_neurons,
                                     params={'window': self.poll_steps})
        self.input_neurons = self.sim.create(parrot_neuron, num_neurons)
        self.sim.connect(drive_view, self.input_neurons, rule=one_to_one, weight=1.0)
        self._drive = drive_view.segments[0].population

        # Motor layer + a single recorder tapping all motor neurons (columns = cells).
        self.motor_neurons = self.sim.create(iaf_psc_exp, num_neurons)
        self.spike_recorder = self.sim.create(spike_recorder)
        self.sim.connect(self.motor_neurons, self.spike_recorder)

        self._spikes = None  # (poll_steps, num_neurons) of the most recent turn

    # -- network construction helpers (used by subclasses) --------------------
    def _set_initial_weights(self, source, target, mean, std):
        """Draw per-edge ``Normal(mean, std)`` pA weights into a static projection.

        Mirrors NEST's ``nest.random.normal`` at connect time, but drawn from the
        network's private RNG (reproducible) and written in place after
        ``reset_rollout`` so the first turn already sees them.
        """
        conns = self.sim.get_connections(source=source, target=target)
        conns.set('weight', self.rng.normal(mean, std, size=len(conns)) * u.pA)

    # -- per-turn host API ----------------------------------------------------
    def set_input_spiketrain(self, input_cell, biological_time=0.0):
        """Clamp a ``N_INPUT_SPIKES``-spike train onto the input neuron ``input_cell``.

        The schedule is turn-local (one row per step); ``biological_time`` is accepted
        for API parity with NEST but not needed — :func:`calculate_stdp` and the
        critic reward are translation-invariant, so the absolute turn offset cancels.
        """
        self.target_index = int(input_cell)
        self.input_train = self.input_t_offset + np.arange(N_INPUT_SPIKES) * ISI
        steps = np.round(self.input_train / DT).astype(int)
        # Spikes scheduled past the turn never fire (NEST rewrites the generator each
        # turn before they would) — clip them, so a large input_t_offset (Dopa's 32 ms)
        # simply drops the train's tail rather than wrapping the schedule counter.
        steps = steps[steps < self.poll_steps]
        schedule = np.zeros((self.poll_steps, self.num_neurons))
        schedule[steps, input_cell] = 1.0
        self._drive.set_schedule(schedule)

    def run_turn(self):
        """Advance the network one 200 ms turn; cache and return the motor spikes."""
        res = self.sim.cont(POLL_TIME * u.ms)
        self._spikes = np.asarray(res.spikes(self.spike_recorder))  # (poll_steps, n)
        return self._spikes

    def get_spike_counts(self):
        """Per-motor-neuron spike count from the most recent turn."""
        return self._spikes.sum(axis=0)

    def get_max_activation(self):
        """Index of the most active motor neuron (random tie-break, like NEST)."""
        spikes = self.get_spike_counts()
        return int(self.rng.choice(np.flatnonzero(spikes == spikes.max())))

    def calculate_reward(self):
        """Reward for the last turn from the winning neuron's distance to the target.

        Reproduces NEST's temporal-difference baseline: the bare distance reward minus
        a per-target running mean, with the mean advanced by half the reward. Records
        the weight matrix and mean-reward snapshots into the performance history.
        """
        self.winning_neuron = self.get_max_activation()
        distance = abs(self.winning_neuron - self.target_index)
        bare_reward = REWARDS_DICT.get(distance, 0.0)
        reward = bare_reward - self.mean_reward[self.target_index]
        self.mean_reward[self.target_index] = float(
            self.mean_reward[self.target_index] + reward / 2.0)

        self.weight_history.append(self.get_all_weights())
        self.mean_reward_history.append(self.mean_reward.copy())
        return reward

    def get_all_weights(self):
        """Dense ``(num_neurons, num_neurons)`` input→motor weight matrix (pA).

        Rows index input (pre) cells, columns motor (post) cells — the layout the host
        STDP update is written against.
        """
        conns = self.sim.get_connections(source=self.input_neurons,
                                         target=self.motor_neurons)
        g = conns.get(['source', 'target', 'weight'])
        weights = np.zeros((self.num_neurons, self.num_neurons))
        weights[np.asarray(g['source']), np.asarray(g['target'])] = _to_pA(g['weight'])
        return weights

    def reset(self):
        """Clear the cached turn (NEST clears its spike recorders here)."""
        self._spikes = None

    def get_performance_data(self):
        """``(mean_reward_history, weight_history)`` across all turns so far."""
        return self.mean_reward_history, self.weight_history

    def apply_synaptic_plasticity(self):
        """Apply the learning rule after a turn (subclass)."""
        raise NotImplementedError


class PongNetRSTDP(PongNetBase):
    """Pong learner with reward-modulated STDP on static input→motor synapses.

    After every turn the host reads the motor spikes, computes a scalar reward from
    the winning neuron's distance to the target cell, and rewrites the static weights
    out of the target input neuron by ``learning_rate · correlation · reward`` (the
    correlation is :func:`calculate_stdp` between the input train and each motor
    neuron's spikes). With ``apply_noise`` a ``noise_generator`` pushes the motor
    neurons over threshold (input only biases which fires); without it the initial
    weights are scaled up to compensate.
    """

    #: Offset (ms) for input spikes in every turn.
    input_t_offset = 1
    #: Learning rate in weight updates.
    learning_rate = 0.7
    #: Amplitude of the STDP curve (arbitrary units).
    stdp_amplitude = 36.0
    #: Time constant of the STDP curve (ms).
    stdp_tau = 64.0
    #: Saturation for the accumulated STDP correlation.
    stdp_saturation = 128
    #: Initial mean weight for input→motor synapses (pA).
    mean_weight = 1300.0

    def __init__(self, apply_noise=True, num_neurons=20, *, seed=0):
        super().__init__(apply_noise, num_neurons, seed=seed)

        if apply_noise:
            self.background_generator = self.sim.create(
                noise_generator, num_neurons, params={'std': BG_STD * u.pA})
            self.sim.connect(self.background_generator, self.motor_neurons,
                             rule=one_to_one)
            init_mean, init_std = self.mean_weight, 1.0
        else:
            # No background noise → compensate by scaling the input weights up so the
            # motor neurons still reach threshold from the input alone.
            init_mean, init_std = self.mean_weight * 1.22, 5.0

        self.sim.connect(self.input_neurons, self.motor_neurons, rule=all_to_all,
                         synapse=static_synapse(weight=self.mean_weight * u.pA))
        self.sim.reset_rollout()
        self._set_initial_weights(self.input_neurons, self.motor_neurons,
                                  init_mean, init_std)

    def apply_synaptic_plasticity(self):
        """Reward the network and apply R-STDP to the target neuron's synapses."""
        reward = self.calculate_reward()
        self.apply_rstdp(reward)

    def apply_rstdp(self, reward):
        """Update the static weights out of the target input neuron by the R-STDP rule.

        For each motor neuron ``j``, ``Δw = learning_rate · calculate_stdp(input_train,
        motor_spikes_j) · reward``; written per-edge in place (no recompile). The edge
        order is read from the collection's own ``target`` array, so the update is
        robust to enumeration order.
        """
        conns = self.sim.get_connections(source=self.input_neurons[self.target_index],
                                         target=self.motor_neurons)
        g = conns.get(['target', 'weight'])
        targets = np.asarray(g['target'])
        old_weight = _to_pA(g['weight'])
        new_weight = old_weight.copy()
        for edge, motor in enumerate(targets):
            motor_spikes = np.where(self._spikes[:, motor] > 0)[0] * DT
            correlation = self.calculate_stdp(self.input_train, motor_spikes)
            new_weight[edge] = old_weight[edge] + self.learning_rate * correlation * reward
        conns.set('weight', new_weight * u.pA)

    def calculate_stdp(self, pre_spikes, post_spikes, only_causal=True, next_neighbor=True):
        """:func:`calculate_stdp` bound to this network's STDP constants."""
        return calculate_stdp(pre_spikes, post_spikes,
                              stdp_amplitude=self.stdp_amplitude, stdp_tau=self.stdp_tau,
                              stdp_saturation=self.stdp_saturation,
                              only_causal=only_causal, next_neighbor=next_neighbor)

    def __repr__(self):
        return ('noisy ' if self.apply_noise else 'clean ') + 'R-STDP'


class PongNetDopa(PongNetBase):
    """Pong learner with dopaminergic plasticity driven by an actor–critic circuit.

    The input→motor synapses are ``stdp_dopamine_synapse`` edges whose weights evolve
    online from a broadcast dopamine concentration ``n(t)`` (a ``volume_transmitter``).
    A *critic* — three ``iaf_psc_exp`` populations (striatum, ventral pallidum, and
    dopaminergic neurons) — turns each turn's performance into that dopamine signal
    (Potjans et al., 2011): the host injects a reward *current* into the dopaminergic
    neurons proportional to the fraction of motor activity at the target cell, and
    their spikes drive the transmitter. The reward current occupies the first
    ``input_t_offset`` ms of each turn (the input spiketrain starts after it).

    Unlike R-STDP the input→motor weights are **not** host-written: they evolve inside
    ``cont()`` from the dopamine signal. The only host write per turn is the reward
    schedule (a fixed-shape :class:`host_current_drive` rewrite — no recompile).

    The initial input→motor / input→striatum weights are set to the mean at connect
    (NEST draws ``Normal(mean, 8)``; the 0.6 % jitter is immaterial and symmetry is
    broken by the poisson background of the noisy variant).
    """

    #: Base reward current applied regardless of performance (pA).
    baseline_reward = 100.0
    #: Maximum reward current applied to the dopaminergic neurons (pA).
    max_reward = 1000
    #: Scaling factor from target-activity fraction to reward current (pA).
    dopa_signal_factor = 4800
    #: Offset (ms) of the input spiketrain — reserves the turn's start for the reward.
    input_t_offset = 32
    #: Initial mean weight for input→motor synapses (pA).
    mean_weight = 1275.0
    #: Standard deviation of the NEST initial-weight draw (recorded; see class doc).
    weight_std = 8
    #: Neurons per critic population.
    n_critic = 8
    #: Weight from striatum / VP to the dopaminergic neurons (pA).
    w_da = -1150
    #: Weight from striatum to VP (pA).
    w_str_vp = -250
    #: Delay (ms) on the direct striatum→dopaminergic connection.
    d_dir = 200
    #: Rate (Hz) of the poisson background generators (noisy variant).
    poisson_rate = 15

    # stdp_dopamine_synapse common properties (NEST ``SetDefaults``); A_minus / tau_minus
    # stay at the NEST-matching spec defaults (1.5 / 20 ms).
    syn_A_plus = 0.85
    syn_tau_plus = 45.0
    syn_tau_c = 70.0
    syn_tau_n = 30.0
    syn_b = 0.028
    syn_Wmin = 1220
    syn_Wmax = 1550

    def __init__(self, apply_noise=True, num_neurons=20, *, seed=0):
        super().__init__(apply_noise, num_neurons, seed=seed)

        self.vt = self.sim.create(volume_transmitter, tau_n=self.syn_tau_n * u.ms)

        if apply_noise:
            motor_mean, motor_wmax = self.mean_weight, self.syn_Wmax
        else:
            # No poisson background → compensate with stronger weights and a wider clamp.
            motor_mean, motor_wmax = self.mean_weight * 1.3, 1750

        self.motor_proj = self.sim.connect(
            self.input_neurons, self.motor_neurons,
            synapse=self._dopa_synapse(motor_mean, motor_wmax), rule=all_to_all, vt=self.vt)

        if apply_noise:
            self.poisson_noise = self.sim.create(
                poisson_generator, num_neurons, params={'rate': self.poisson_rate * u.Hz})
            self.sim.connect(self.poisson_noise, self.motor_neurons, rule=one_to_one,
                             weight=self.mean_weight * u.pA)

        # Critic: striatum -> VP -> dopaminergic neurons -> volume_transmitter.
        self.striatum = self.sim.create(iaf_psc_exp, self.n_critic)
        self.sim.connect(self.input_neurons, self.striatum,
                         synapse=self._dopa_synapse(self.mean_weight, self.syn_Wmax),
                         rule=all_to_all, vt=self.vt)
        self.vp = self.sim.create(iaf_psc_exp, self.n_critic)
        self.sim.connect(self.striatum, self.vp, rule=all_to_all, weight=self.w_str_vp * u.pA)
        self.dopa = self.sim.create(iaf_psc_exp, self.n_critic)
        self.sim.connect(self.vp, self.dopa, rule=all_to_all, weight=self.w_da * u.pA)
        self.sim.connect(self.striatum, self.dopa, rule=all_to_all,
                         weight=self.w_da * u.pA, delay=self.d_dir * u.ms)
        self.sim.connect(self.dopa, self.vt)

        # Host-set reward current into the dopaminergic neurons (first input_t_offset ms).
        reward_view = self.sim.create(host_current_drive, self.n_critic,
                                      params={'window': self.poll_steps})
        self.sim.connect(reward_view, self.dopa, rule=one_to_one)
        self._reward_drive = reward_view.segments[0].population
        self._offset_steps = int(round(self.input_t_offset / DT))

        self.sim.reset_rollout()

    def _dopa_synapse(self, weight, wmax):
        """An ``stdp_dopamine_synapse`` spec with NEST's PongNetDopa common properties."""
        return stdp_dopamine_synapse(
            weight=weight * u.pA, A_plus=self.syn_A_plus,
            tau_plus=self.syn_tau_plus * u.ms, tau_c=self.syn_tau_c * u.ms,
            tau_n=self.syn_tau_n * u.ms, b=self.syn_b, Wmin=self.syn_Wmin, Wmax=wmax)

    def apply_synaptic_plasticity(self):
        """Inject a reward current into the dopaminergic neurons for the next turn.

        The current is proportional to the fraction of the last turn's motor activity
        at the target neuron (clipped to ``max_reward``), scheduled across the first
        ``input_t_offset`` ms of the next turn. The dopamine-driven weight change itself
        happens inside ``cont()``; this only sets the signal. :meth:`calculate_reward`
        is then called for the performance metrics (and to pick the winning neuron).
        """
        counts = self.get_spike_counts()
        target_n_spikes = counts[self.target_index]
        total_n_spikes = max(int(counts.sum()), 1)
        reward_current = self.dopa_signal_factor * target_n_spikes / total_n_spikes \
            + self.baseline_reward
        reward_current = min(reward_current, self.max_reward)

        schedule = np.zeros((self.poll_steps, self.n_critic))
        schedule[:self._offset_steps, :] = reward_current
        self._reward_drive.set_schedule(schedule)

        self.calculate_reward()

    def __repr__(self):
        return ('noisy ' if self.apply_noise else 'clean ') + 'TD'
