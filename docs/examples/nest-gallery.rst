NEST-Compatible Examples
========================

A **Rosetta map** of PyNEST examples ported to ``brainpy.state``. Each entry
keeps NEST's model and parameter names, so a NEST user can find the familiar
example on the left and the ``brainpy.state`` equivalent on the right. All
scripts live in the `examples/nest_like/
<https://github.com/chaobrain/brainpy.state/tree/main/examples/nest_like>`_
directory and run end to end through the :class:`~brainpy.state.Simulator` (one
compiled ``for_loop`` — no Python step loop).

Every port is backed by a live-NEST parity test where a sample-for-sample
comparison is meaningful; see :doc:`/nest-style/validation-status` for the
per-model parity evidence and tolerance bands, and the
:doc:`/nest-style/index` hub for the build-your-network guides.

.. admonition:: Reading the map
   :class: note

   Each line reads ``script.py`` — *NEST upstream* → ``brainpy.state`` model or
   feature — one-line summary. Where an upstream mechanism is NEST-specific
   (e.g. the Connection Set Algebra), the port re-expresses the *connectivity it
   describes* natively; those entries are marked.

First steps: single & two neurons
---------------------------------

The PyNEST "Part 1-2" first-contact examples — drive one or two neurons and
record them.

``one_neuron.py``
    NEST ``one_neuron`` → ``iaf_psc_alpha`` + ``voltmeter``. A single neuron
    driven by a constant ``I_e``, observed by a reverse-connected voltmeter.

``one_neuron_with_noise.py``
    NEST ``one_neuron_with_noise`` → ``iaf_psc_alpha`` + 2-channel
    ``poisson_generator``. Excitatory/inhibitory Poisson drive with signed
    per-channel weights.

``twoneurons.py``
    NEST ``twoneurons`` → two ``iaf_psc_alpha`` + ``static_synapse``. A driven
    presynaptic neuron connected to a postsynaptic neuron, each with its own
    voltmeter.

``testiaf.py``
    NEST ``testiaf`` → ``iaf_psc_alpha`` swept over ``dt`` ∈ {0.1, 0.5, 1.0} ms.
    Constant-current charge / spike / refractory / recovery (rebuild-per-trial).

``if_curve.py``
    NEST ``if_curve`` → ``aeif_cond_exp`` + ``noise_generator``. Population firing
    rate across an ``(I_mean, I_std)`` grid — the neuron's transfer function.

``vinit_example.py``
    NEST ``vinit_example`` → ``iaf_cond_exp_sfa_rr``. Passive relaxation toward
    ``E_L`` from several initial membrane voltages (rebuild-per-trial).

``balancedneuron.py``
    NEST ``balancedneuron`` → ``iaf_psc_alpha`` + 2-channel ``poisson_generator``.
    Root-find the inhibitory rate that balances the neuron's output rate
    (SciPy ``bisect``, rebuild-per-trial).

``izhikevich.py``
    NEST ``izhikevich`` → ``izhikevich`` neuron. The canonical Izhikevich (2003)
    firing regimes.

``hh_psc_alpha.py``
    NEST ``hh_psc_alpha`` → ``hh_psc_alpha`` + ``multimeter`` + ``spike_recorder``.
    Hodgkin-Huxley F-I curve from a constant-current sweep.

``hh_phaseplane.py``
    NEST ``hh_phaseplane`` → ``hh_psc_alpha`` (analysis). V-n phase-plane vector
    field, nullclines, and a limit cycle — a pedagogical analysis demo.

Neuron models
-------------

Single-neuron ports exercising specific neuron families — adaptive exponential,
generalized-LIF, multi-timescale, and multi-compartment models.

``aeif_cond_beta_multisynapse.py``
    NEST ``aeif_cond_beta_multisynapse`` → same model. One spike fans out to four
    distinct beta-function conductance receptors via ``connect(receptor_type=k)``.

``brette_gerstner_fig_2c.py``
    NEST ``brette_gerstner_fig_2c`` → ``aeif_cond_alpha``. AdEx spike-frequency
    adaptation (Brette & Gerstner 2005, Fig 2C).

``brette_gerstner_fig_3d.py``
    NEST ``brette_gerstner_fig_3d`` → ``aeif_cond_exp``. AdEx post-inhibitory
    rebound burst (Fig 3D).

``gif_cond_exp_multisynapse.py``
    NEST ``gif_cond_exp_multisynapse`` → same model. Generalized-IAF neuron with
    two exponential-conductance receptors.

``glif_cond_neuron.py``
    NEST ``glif_cond_neuron`` → ``glif_cond``. Conductance-based Allen-Institute
    GLIF at its five mechanism levels (RKF45 conductance synapses).

``glif_psc_neuron.py``
    NEST ``glif_psc_neuron`` → ``glif_psc``. Current-based GLIF at five mechanism
    levels, advanced with exact propagator matrices.

``glif_psc_double_alpha_neuron.py``
    NEST ``glif_psc_double_alpha_neuron`` → ``glif_psc_double_alpha``. Double-alpha
    post-synaptic currents across three receptor ports.

``mat_psc_exp.py``
    NEST ``mat_psc_exp`` → ``mat2_psc_exp`` / ``amat2_psc_exp``. Multi-timescale
    adaptive-threshold neurons whose membrane never resets.

``mc_neuron.py``
    NEST ``mc_neuron`` → ``iaf_cond_alpha_mc``. Three-compartment neuron with nine
    receptors selecting compartment and channel.

``two_comps.py``
    NEST compartmental ``two_comps`` → ``cm_default``. Two-compartment models with
    active vs passive dendrites and attached AMPA/GABA/NMDA receptors.

``receptors_and_current.py``
    NEST compartmental ``receptors_and_current`` → ``cm_default``. A soma + two
    dendrites with a different receptor per compartment plus a current injection.

``wang_decision_making.py``
    NEST ``wang_decision_making`` → ``iaf_bw_2001``. Wang (2002) spiking
    decision-making network with AMPA/GABA/recurrent-NMDA receptors.

Balanced & Brunel networks
--------------------------

Random balanced networks — the Brunel family and related E/I benchmarks.

``brunel_alpha.py``
    NEST ``brunel_alpha_nest`` → ``iaf_psc_alpha`` + ``poisson_generator`` +
    ``spike_recorder``. Brunel (2000) balanced random network with alpha synapses.

``brunel_delta.py``
    NEST ``brunel_delta_nest`` → ``iaf_psc_delta``. Brunel network with delta
    synapses (instantaneous voltage jumps; weights in mV).

``brunel_exp_multisynapse.py``
    NEST ``brunel_exp_multisynapse_nest`` → ``iaf_psc_exp_multisynapse``. Brunel
    network with per-connection uniformly-drawn receptor ports.

``brunel_alpha_evolution_strategies.py``
    NEST ``brunel_alpha_evolution_strategies`` → ``iaf_psc_alpha`` (Brunel alpha).
    Natural-evolution-strategies tuning of ``eta`` and ``g`` to target statistics.

``brunel_siegert.py``
    NEST ``brunel_siegert_nest`` → ``siegert_neuron`` + ``diffusion_connection``.
    Mean-field (Siegert) rates for the Brunel network via pseudo-time relaxation.

``brette_et_al_2007.py``
    NEST ``brette_et_al_2007/{coba,cuba}`` → ``iaf_cond_exp`` / ``iaf_psc_exp``.
    The Vogels-Abbott COBA + CUBA FACETS review benchmarks.

``ei_clustered_network.py``
    NEST ``EI_clustered_network`` → ``iaf_psc_exp``. Clustered random balanced
    network with potentiated in-cluster / depressed out-cluster weights.

``sensitivity_to_perturbation.py``
    NEST ``sensitivity_to_perturbation`` → ``iaf_psc_delta`` (Brunel-style).
    Two trials differing by one spike show chaotic decorrelation.

``artificial_synchrony.py``
    NEST ``artificial_synchrony`` (grid branch) → ``iaf_psc_alpha``. Grid-induced
    artificial synchrony measured by the Golomb-Rinzel statistic.

Devices: generators, recorders & detectors
------------------------------------------

The device family — stimulation generators, recorders/multimeters, and
correlation detectors (see :doc:`/nest-style/devices`).

``multimeter_file.py``
    NEST ``multimeter_file`` → ``multimeter`` (in-memory). Recording several
    analog variables; read back with ``res.trace(mm, name)``.

``recording_demo.py``
    NEST ``recording_demo`` → ``poisson_generator`` + ``iaf_psc_exp`` +
    recorders. A tour of the recording API.

``repeated_stimulation.py``
    NEST ``repeated_stimulation`` → ``poisson_generator`` gated to a
    ``[start, stop]`` window, repeated across trials.

``BrodyHopfield.py``
    NEST ``BrodyHopfield`` → ``iaf_psc_alpha`` + ``ac_generator`` +
    ``noise_generator``. Spike synchronization to a weak subthreshold oscillation.

``sinusoidal_poisson_generator.py``
    NEST ``sinusoidal_poisson_generator`` → same device. An inhomogeneous Poisson
    train with a sinusoidally-modulated instantaneous rate.

``sinusoidal_gamma_generator.py``
    NEST ``sinusoidal_gamma_generator`` → same device. A sinusoidally-modulated
    gamma-process drive (regularity controlled by the shape parameter).

``pulsepacket.py``
    NEST ``pulsepacket`` → ``pulsepacket_generator``. Gaussian-profile synchronous
    spike volleys and their effect on a target neuron.

``precise_spiking.py``
    NEST ``precise_spiking`` → ``iaf_psc_exp`` vs ``iaf_psc_exp_ps``. Grid vs
    precise (off-grid) spike timing across resolutions.

``correlospinmatrix_detector_two_neuron.py``
    NEST ``correlospinmatrix_detector_two_neuron`` → ``ginzburg_neuron`` +
    ``mcculloch_pitts_neuron`` + ``correlospinmatrix_detector``. Binary
    auto-/cross-covariances.

``cross_check_mip_corrdet.py``
    NEST ``cross_check_mip_corrdet`` → ``mip_generator`` + ``correlation_detector``.
    A cross-correlogram cross-checked against a hand-written reference.

``CampbellSiegert.py``
    NEST ``CampbellSiegert`` → ``iaf_psc_alpha`` (analytic cross-check). Campbell's
    theorem + Siegert's approximation vs a simulated population.

Plasticity: STP, STDP & dendritic rules
---------------------------------------

Short-term plasticity, spike-timing-dependent plasticity, and dendritic
prediction-error learning (see :doc:`/nest-style/divergences/stdp`).

``evaluate_tsodyks2_synapse.py``
    NEST ``evaluate_tsodyks2_synapse`` → ``tsodyks2_synapse``. Depressing and
    facilitating short-term-plasticity envelopes read off a linear post neuron.

``evaluate_quantal_stp_synapse.py``
    NEST ``evaluate_quantal_stp_synapse`` → ``quantal_stp_synapse``. The
    stochastic (multi-release-site) Tsodyks-Markram variant converging to the
    deterministic envelope.

``iaf_tum_2000_short_term_depression.py``
    NEST ``iaf_tum_2000_short_term_depression`` → ``iaf_tum_2000``. Presynaptic
    short-term depression (Tsodyks et al. 1998, Fig 1A).

``iaf_tum_2000_short_term_facilitation.py``
    NEST ``iaf_tum_2000_short_term_facilitation`` → ``iaf_tum_2000``. Presynaptic
    short-term facilitation (Fig 1B).

``clopath_synapse_spike_pairing.py``
    NEST ``clopath_synapse_spike_pairing`` → ``aeif_psc_delta_clopath`` +
    ``clopath_synapse``. Voltage-based STDP weight change vs pairing frequency.

``clopath_synapse_small_network.py``
    NEST ``clopath_synapse_small_network`` → ``aeif_psc_delta_clopath`` +
    ``clopath_synapse``. Clopath rule establishing directional structure
    (made deterministic).

``urbanczik_synapse_example.py``
    NEST ``urbanczik_synapse_example`` → ``pp_cond_exp_mc_urbanczik`` +
    ``urbanczik_synapse``. Dendritic prediction-error plasticity (Urbanczik &
    Senn 2014).

Gap junctions
-------------

Electrical coupling via the one-step-lagged difference-current scheme.

``gap_junctions_two_neurons.py``
    NEST ``gap_junctions_two_neurons`` → ``hh_psc_alpha_gap`` + ``gap_junction``.
    Two cells synchronizing through a single symmetric electrical coupling.

``gap_junctions_inhibitory_network.py``
    NEST ``gap_junctions_inhibitory_network`` → ``hh_psc_alpha_gap``. An
    inhibitory network whose synchrony rises with the gap-junction weight
    (Hahne et al. 2015).

Astrocytes & tripartite networks
--------------------------------

Neuron-astrocyte (tripartite) models using ``tripartite_connect`` and
``sic_connection``.

``astrocyte_single.py``
    NEST ``astrocyte_single`` → ``astrocyte_lr_1994`` (+ readout neuron).
    A single Poisson-driven astrocyte's IP3/calcium dynamics.

``astrocyte_interaction.py``
    NEST ``astrocyte_interaction`` → ``aeif_cond_alpha_astro`` +
    ``astrocyte_lr_1994`` + ``sic_connection``. The two-neuron + one-astrocyte
    tripartite loop.

``astrocyte_small_network.py``
    NEST ``astrocyte_small_network`` → tripartite ``pairwise_bernoulli`` +
    ``third_factor_bernoulli_with_pool``. A small neuron-astrocyte network.

``astrocyte_brunel_bernoulli.py``
    NEST ``astrocyte_brunel_bernoulli`` → Brunel + astrocytes, Bernoulli primary
    wiring with a random astrocyte pool.

``astrocyte_brunel_fixed_indegree.py``
    NEST ``astrocyte_brunel_fixed_indegree`` → Brunel + astrocytes, fixed-indegree
    primary wiring (shares the Bernoulli assembly).

Rate models & mean-field
------------------------

Rate-based neurons and mesoscopic population-rate models.

``lin_rate_ipn_network.py``
    NEST ``lin_rate_ipn_network`` → ``lin_rate_ipn``. Linear rate-neuron network
    with delayed excitation and instantaneous inhibition.

``rate_neuron_dm.py``
    NEST ``rate_neuron_dm`` → ``lin_rate_ipn``. A two-unit rectified winner-take-all
    rate decision circuit.

``gif_pop_psc_exp.py``
    NEST ``gif_pop_psc_exp`` → ``gif_pop_psc_exp`` vs microscopic ``gif_psc_exp``.
    Mesoscopic population-rate model against its spiking realization.

``gif_population.py``
    NEST ``gif_population`` → ``gif_psc_exp``. An adaptation-driven oscillating GIF
    population under Poisson drive.

Spatially-structured networks
-----------------------------

Distance-dependent connectivity from the ``brainpy.state.spatial`` family (see
:doc:`/nest-style/spatial`).

``spatial_grid_iaf.py``
    NEST ``spatial/grid_iaf`` → ``spatial.grid`` + ``iaf_psc_alpha``. A 4x3 grid
    layer; coordinates read back with ``get_position``.

``spatial_gaussex.py``
    NEST ``spatial/gaussex`` → distance-dependent pairwise-Bernoulli with a
    Gaussian kernel between two grid populations.

``spatial_3d_gauss.py``
    NEST ``spatial/test_3d_gauss`` → ``spatial.free`` 3D layer + Gaussian kernel
    clipped to a cubic mask.

``spatial_gabor.py``
    NEST-like anisotropic kernel → ``gabor`` distance distribution + rotated
    ``elliptical`` mask (per-axis ``distance.x`` / ``distance.y``).

``spatial_csa.py``
    NEST ``csa_spatial_example`` → native distance-dependent pairwise-Bernoulli.
    The Connection Set Algebra expression re-expressed without ``libneurosim``.

``csa_example.py``
    NEST ``csa_example`` → native ``pairwise_bernoulli`` equivalent. The CSA /
    ``conngen`` mechanism is intentionally not ported; the connectivity it
    describes is shown natively.

Connectivity & introspection
----------------------------

Building, retrieving, and inspecting connections (see
:doc:`/nest-style/connectivity`).

``synapsecollection.py``
    NEST ``synapsecollection`` → ``Simulator.get_connections``. Connecting under
    several rules/synapse models and reading/setting ``.source`` / ``.target`` /
    ``.weight``.

``plot_weight_matrices.py``
    NEST ``plot_weight_matrices`` → connection introspection. Extract every
    realized weight and assemble the EE/EI/IE/II matrices for visualization.

Applications: learning & large networks
---------------------------------------

Larger, application-flavored networks — reinforcement learning and constraint
solving. These are the "large networks" entries featured on the
:doc:`/nest-style/index` hub.

``pong.py``
    NEST ``pong/pong`` → pure-Python game logic. The Pong playing field, ball,
    and paddles used by the spiking learners.

``pong_networks.py``
    NEST ``pong/networks`` → ``iaf_psc_exp`` learners. Reward-modulated STDP
    (R-STDP) and dopaminergic spiking networks that learn the input→output map.

``pong_run.py``
    NEST ``pong/run_simulations`` → harness. Trains the two spiking learners
    against each other turn by turn with ``Simulator.cont``.

``sudoku.py``
    NEST ``sudoku/sudoku_solver`` → harness. Relaxes a Sudoku puzzle with a
    spiking winner-take-all network, reading out per-cell winners each chunk.

``sudoku_net.py``
    NEST ``sudoku/sudoku_net`` → ``iaf_psc_exp`` WTA network. Inhibitory
    connectivity encoding the row/column/box/cell Sudoku constraints.

``sudoku_puzzles.py``
    NEST ``sudoku/helpers_sudoku`` → pure-Python puzzle bank + solution validator
    (NumPy only).

See Also
--------

- :doc:`/nest-style/index` — the NEST-compatible hub (tutorials, build-your-network
  guides, porting & fidelity).
- :doc:`/nest-style/validation-status` — per-model live-NEST parity evidence.
- :doc:`/examples/brainpy-gallery` — the BrainPy-style example gallery.
