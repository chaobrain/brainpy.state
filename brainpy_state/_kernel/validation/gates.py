# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Registry of the K0 go/no-go gates (spec ``2026-10-21-k0-kernel-spec.md`` §6).

Each :class:`Gate` lists the test node-id prefixes (relative to
``brainpy_state/_kernel/``) that implement it. ``brainpy_state/_kernel/conftest.py``
attaches the ``k0_gate`` marker and the per-gate marker (``k0_g1`` ... ``k0_g8``) to
every collected test matching a prefix, so the gate tests written by T3--T9 are
selected where they live instead of being duplicated::

    pytest -m k0_gate brainpy_state/_kernel          # every gate test
    pytest -m k0_g2 brainpy_state/_kernel            # one gate

Gates without a pytest implementation carry a ``command`` instead (G9 performance on
the remote host, G10 coverage). Gates marked ``requires`` need live NEST or Brian2
(skipped otherwise).
"""
from __future__ import annotations

import dataclasses
from typing import Optional

__all__ = ['Gate', 'GATES', 'gates_for', 'marker_names']


@dataclasses.dataclass(frozen=True)
class Gate:
    """One gate of spec §6.

    Attributes
    ----------
    id : str
        ``'G1'`` ... ``'G10'``.
    title : str
        Short name.
    criterion : str
        Pass criterion as tested.
    tests : tuple of str
        Node-id prefixes relative to ``brainpy_state/_kernel/`` (``file``,
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
        """Per-gate pytest marker name (``k0_g1`` ...)."""
        return 'k0_' + self.id.lower()


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
         'detached reset and membrane loss; relaxed_spikes() recurrent)',
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
    Gate('G10', 'coverage of brainpy_state/_kernel', '> 90 %',
         command='python -m pytest -n 8 brainpy_state/_kernel --cov=brainpy_state/_kernel '
                 '--cov-report=term'),
)


def gates_for(rel_nodeid: str) -> list:
    """Gates whose test prefixes match a node id relative to ``brainpy_state/_kernel/``.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy_state._kernel.validation.gates import gates_for
        >>> [g.id for g in gates_for('learn/bptt_test.py::test_g6_relaxed_vs_finite_differences[x]')]
        ['G6']
        >>> gates_for('units_test.py::test_x')
        []
    """
    return [g for g in GATES if any(rel_nodeid.startswith(p) for p in g.tests)]


def marker_names() -> list:
    """``['k0_gate', 'k0_g1', ...]`` for marker registration."""
    return ['k0_gate'] + [g.marker for g in GATES]
