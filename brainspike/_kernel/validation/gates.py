# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Registry of the K0 go/no-go gates (spec ``2026-10-21-k0-kernel-spec.md`` §6) and
of the K1a gates (spec ``2026-10-22-k1a-kernel-hardening-spec.md`` §K1a-7).

Each :class:`Gate` lists the test node-id prefixes (relative to
``brainspike/_kernel/``) that implement it. ``brainspike/_kernel/conftest.py``
attaches the ``k0_gate`` marker and the per-gate marker (``k0_g1`` ... ``k0_g8``) to
every collected test matching a prefix, so the gate tests written by T3--T9 are
selected where they live instead of being duplicated::

    pytest -m k0_gate brainspike/_kernel          # every gate test
    pytest -m k0_g2 brainspike/_kernel            # one gate

Gates without a pytest implementation carry a ``command`` instead (G9 performance on
the remote host, G10 coverage). Gates marked ``requires`` need live NEST or Brian2
(skipped otherwise).

The K1a gates (:data:`K1A_GATES`, ids ``K1a-G1`` ... ``K1a-G6``) work the same way with
the ``k1a_gate`` marker and the per-gate markers ``k1a_g1`` ... ``k1a_g6``::

    pytest -m "k0_gate or k1a_gate" brainspike/_kernel   # the CI gate job

Their benchmark parts (compile time, step time, memory, the full-scale microcircuit)
are commands run on the remote host; the pytest parts are correctness checks.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

__all__ = ['Gate', 'GATES', 'K1A_GATES', 'ALL_GATES', 'gates_for', 'marker_names']


@dataclasses.dataclass(frozen=True)
class Gate:
    """One gate of the K0 spec §6 or the K1a spec §K1a-7.

    Attributes
    ----------
    id : str
        ``'G1'`` ... ``'G10'`` (K0) or ``'K1a-G1'`` ... ``'K1a-G6'`` (K1a).
    title : str
        Short name.
    criterion : str
        Pass criterion as tested.
    tests : tuple of str
        Node-id prefixes relative to ``brainspike/_kernel/`` (``file``,
        ``file::Class``, ``file::Class::test`` or ``file::test_prefix``).
    command : str or None
        Command for gates that are not (only) pytest tests.
    requires : str
        ``''``, ``'nest'``, ``'brian2'`` or ``'bench'``.
    """
    id: str
    title: str
    criterion: str
    tests: tuple = ()
    command: Optional[str] = None
    requires: str = ''

    @property
    def marker(self) -> str:
        """Per-gate pytest marker name (``k0_g1`` ..., ``k1a_g1`` ...)."""
        if self.id.startswith('K1a-'):
            return 'k1a_' + self.id[4:].lower()
        return 'k0_' + self.id.lower()

    @property
    def suite(self) -> str:
        """Suite marker (``k0_gate`` or ``k1a_gate``)."""
        return 'k1a_gate' if self.id.startswith('K1a-') else 'k0_gate'


GATES: tuple = (
    Gate('G1', 'timing per port kind vs NEST (align_steps = 0)',
         'exact models CAT_B; iaf_cond_exp max |dV_m| <= 1e-3 mV and equal spike times; '
         'precise spike times within 1e-6 ms',
         tests=(
             'network_nest_test.py::TestNetworkDeliveryVsNest',          # event + current (T5)
             'models/nest_parity_test.py::TestIafPscAlphaVsNest',        # single-neuron (T3)
             'validation/g1_timing_nest_test.py',                        # conductance (T10)
             'models/precise_nest_test.py::TestPreciseVsNest',           # timed_event (T7)
         ), requires='nest'),
    Gate('G2', 'Brunel iaf_psc_alpha order 200 statistics vs NEST',
         '|dmean| <= max(5 % of NEST mean, 2 x combined SEM), 10 seeds each side',
         tests=('validation/g2_brunel_nest_test.py',), requires='nest'),
    Gate('G3', 'COBA Vogels-Abbott vs Brian2',
         'network statistics by the G2 rule; single-neuron traces CAT_A',
         tests=('validation/g3_coba_brian2_test.py',), requires='brian2'),
    Gate('G4', 'invariance and goal-01 repros', 'exact equality',
         tests=(
             'connectivity/invariance_test.py',
             'connectivity/rules_test.py::TestAllToAll::test_autapses_on_disjoint_slices',
             'connectivity/rules_test.py::TestAllToAll::test_autapses_on_overlapping_slices',
             'connectivity/rules_test.py::TestAllToAll::test_concatenated_views',
             'connectivity/rules_test.py::TestFixedIndegree::test_exact_indegree_no_autapses_on_overlap',
             'connectivity/rules_test.py::TestFixedIndegree::test_concatenated_pre_is_one_population',
             'connectivity/rules_test.py::TestFixedIndegree::test_x64_invariance',
             'connectivity/rules_test.py::TestPairwiseBernoulli::test_block_size_invariance',
             'connectivity/rules_test.py::TestFixedTotalNumber::test_exact_count_no_autapses',
             'connectivity/edgeset_test.py::TestGetConnections::test_zero_weights_and_multapses_kept',
             'connectivity/edgeset_test.py::TestSet::test_set_aligned_subset',
             'rng_test.py::TestTrialKeys',
             'runner_test.py::TestBatchingAndRng',
             'runner_test.py::TestViewsEndToEnd',
             'runner_test.py::TestDelayBuffers::test_post_side_equals_pre_side_for_equal_delays',
             'models/devices_test.py::TestSpikeGenerator::test_duplicates_counted',
             'models/devices_test.py::TestSpikeGenerator::test_multiplicities_and_ragged_per_unit',
             'models/nest_parity_test.py::TestSpikeGeneratorMultiplicityVsNest',
             'validation/g4_invariance_test.py',
         )),
    Gate('G5', 'integrators vs NEST at substeps {1, 2, 4, 8, 16}',
         'convergence at the integrator order; default substeps meet CAT_A (aeif: spike '
         'times equal, V_m CAT_A off the upswing/reset steps)',
         tests=('models/nest_parity_test.py::TestConductanceModelsVsNest',
                'integrators_test.py::TestConvergenceOrder'), requires='nest'),
    Gate('G6', 'BPTT gradients vs central finite differences',
         'relative error <= 1e-4 where BPTT is exact (hard-forward feed-forward with '
         "detached reset and membrane loss; Program(spikes='relaxed') recurrent)",
         tests=('learn/bptt_test.py::test_g6_',)),
    Gate('G7', 'braintrace compatibility',
         'eprop / d_rtrl compile, non-zero gradient for every weight; checker rejects '
         'non-compliant models with a clear error',
         tests=('learn/etrace_test.py::test_g7_',
                'learn/etrace_test.py::test_braintrace_compile_directly_on_network',
                'learn/etrace_test.py::test_checker_',
                'learn/etrace_test.py::test_plan_level_rejections')),
    Gate('G8', 'stdp_synapse vs NEST with dendritic delay', 'CAT_B without the drive workaround',
         tests=('validation/stdp_nest_test.py',), requires='nest'),
    Gate('G9', 'performance vs the old Simulator (remote host, CPU)',
         'median step time <= 1.1 x and peak RSS <= 1.1 x the old Simulator, order 2500, '
         'float32 and float64, interleaved on one idle pinned core (GPU skipped by user '
         'decision)',
         command="/tmp/k0/remote.sh T10 'python benchmarks/g9_interleaved.py run "
                 "--commit <sha> --rounds 3' > g9.log; "
                 "python benchmarks/g9_interleaved.py compare g9.log",
         requires='bench'),
    Gate('G10', 'coverage of brainspike/_kernel', '> 90 %',
         command='python -m pytest -n 8 brainspike/_kernel --cov=brainspike/_kernel '
                 '--cov-report=term'),
)


K1A_GATES: tuple = (
    Gate('K1a-G1', 'compile time of every K0 network at default substeps (CPU)',
         '<= 5 s each (first simulate minus a warm cont, fresh process)',
         tests=('validation/compile_gate_test.py',),
         command="BENCH=1 /tmp/k0/remote.sh K1a-7 'python -m pytest -m k1a_compile "
                 "brainspike/_kernel/validation/compile_gate_test.py'; "
                 "python benchmarks/k1a_compile_time.py all",
         requires='bench'),
    Gate('K1a-G2', 'single-op per-edge delays (K1a-4 gate)',
         'step <= 1.5 x and build <= 2 x the scalar-delay network (2000 neurons, '
         'indegree 100, delays up to 2 / 5 / 20 ms; CPU and GPU, f32 and f64)',
         tests=('delay_scatter_test.py', 'delay_scatter_network_test.py'),
         command='python benchmarks/k1a4_delays.py sweep --net review',
         requires='bench'),
    Gate('K1a-G3', 'G9 on GPU 1 (Brunel order 200 and 2500)',
         'step time and peak device memory <= 1.1 x the old Simulator (f32 default '
         'brainevent backend, f64 cusparse); compile time reported',
         command="GPU=1 /tmp/k0/remote.sh K1a-7 'python benchmarks/k1a_gpu_g9.py run ...'",
         requires='bench'),
    Gate('K1a-G4', 'Potjans-Diesmann microcircuit at full scale vs NEST',
         'per-population rates by the G2 rule, 5 seeds each side, 1 s after 100 ms; '
         'memory / build / compile / step reported (CPU and GPU 1, float32)',
         tests=('validation/microcircuit_test.py', 'validation/microcircuit_nest_test.py',
                'models/iaf_psc_exp_test.py', 'models/nest_parity_test.py::TestIafPscExpVsNest'),
         command='python benchmarks/k1a_microcircuit.py kernel|nest ...',
         requires='bench'),
    Gate('K1a-G5', 'G2 in the asynchronous-irregular regime vs NEST',
         'G2 rule on rate, CV-ISI, Fano factor and correlation; Brunel order 2500, g = 16, '
         'eta = 4 (lead choice from the NEST-only scan), 10 seeds each side',
         tests=('validation/k1a_g5_ai_nest_test.py',), requires='nest'),
    Gate('K1a-G6', 'regressions: every K0 gate', 'green',
         command='python -m pytest -n 8 -m k0_gate brainspike/_kernel'),
)

ALL_GATES: tuple = GATES + K1A_GATES


def gates_for(rel_nodeid: str) -> list:
    """Gates whose test prefixes match a node id relative to ``brainspike/_kernel/``.

    Examples
    --------
    .. code-block:: python

        >>> from brainspike._kernel.validation.gates import gates_for
        >>> [g.id for g in gates_for('learn/bptt_test.py::test_g6_relaxed_vs_finite_differences[x]')]
        ['G6']
        >>> gates_for('units_test.py::test_x')
        []
    """
    return [g for g in ALL_GATES if any(rel_nodeid.startswith(p) for p in g.tests)]


def marker_names() -> list:
    """``['k0_gate', 'k0_g1', ..., 'k1a_gate', 'k1a_g1', ...]`` for marker registration."""
    return (['k0_gate'] + [g.marker for g in GATES]
            + ['k1a_gate'] + [g.marker for g in K1A_GATES])
