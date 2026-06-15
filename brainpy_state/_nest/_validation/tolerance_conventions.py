# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Tolerance conventions for live-NEST parity — the single source of truth.

What tolerance for what kind of model. Categories A-E span the comparison space
the parity harness must cover (V_m trace, firing rate, weight trajectory,
PSC-amplitude train, F-I curve):

==== ============================== ===================== ================ ==============
Cat  Kind                           Metric                Tolerance        Mode
==== ============================== ===================== ================ ==============
A    adaptive numerical integrator  aeif/HH/izh V_m       atol 1e-3 mV     trace
B    analytic exact propagator      linear iaf_psc V_m    atol 1e-6 mV     trace
C    conductance / coupled          iaf_cond / Siegert    1e-3 mV / 5 %    trace
D    distributional (PRNG-diverge)  network rate, ISI CV  mean 5 % (>=4 s) distributional
E    spike-time / event-count       ``*_ps``, PSC timing  |dN|<=2,|dt|<=1  trace
==== ============================== ===================== ================ ==============

The A/B/C numbers come from ``docs/nest-status/internal/numerical-validation-gap.md``
section 6. ``CAT_B_ALIGNED`` is the same analytic family but tolerates a one-step
multimeter offset (``align_steps=1``). ``CAT_C_RATE`` is the deterministic
mean-field rate fixed point (Hz, compared purely relatively). ``D`` numbers the
unnumbered "distributional" default; ``E`` covers precise spiking.

**Multi-seed protocol (category D).** NEST and JAX draw from independent PRNG
streams, so PRNG-divergent drives are never compared per-sample. Run
``N >= N_SEEDS_DEFAULT`` seeds per side and compare the seed-**mean** (legacy
single-realization tests use one seed, a documented limitation).

Notes
-----
``TraceTolerance.atol`` is unit-aware (a :mod:`brainunit` ``Quantity`` in mV) for
voltage traces, and a plain ``float`` for dimensionless / rate metrics. The
``compare_*`` engine in :mod:`brainpy_state._nest._validation.nest_compare`
consumes these constants; this module holds no logic.

Examples
--------
.. code-block:: python

    >>> import brainunit as u
    >>> from brainpy_state._nest._validation import tolerance_conventions as tc
    >>> u.get_unit(tc.CAT_A.atol) == u.mV
    True
    >>> tc.CAT_D.rate_rtol, tc.CAT_D.n_seeds
    (0.05, 5)
"""
import dataclasses

import brainunit as u

__all__ = [
    "TraceTolerance", "DistributionalTolerance", "SpikeTimeTolerance",
    "CAT_A", "CAT_B", "CAT_B_ALIGNED", "CAT_C", "CAT_C_RATE", "CAT_D", "CAT_E",
    "T_DEFAULT", "DT_DEFAULT", "N_SEEDS_DEFAULT",
]


@dataclasses.dataclass(frozen=True)
class TraceTolerance:
    """Per-sample / max-abs tolerance for a deterministic trace (categories A, B, C).

    Parameters
    ----------
    atol : float or brainunit.Quantity
        Absolute tolerance. Unit-aware for voltage traces (e.g. ``1e-3 * u.mV``);
        a plain ``float`` for dimensionless / rate metrics. The pass test is the
        numpy-allclose form ``|a - b| <= atol + rtol * |reference|`` (division-free,
        so a zero reference never divides).
    rtol : float
        Relative tolerance (dimensionless).
    align_steps : int, optional
        Allow a ``+/- align_steps`` integer-sample shift search to absorb a recorder
        one-step offset. Default ``0`` (exact alignment).
    label : str, optional
        Category label (``"A"``/``"B"``/``"C"``) for diagnostics.
    note : str, optional
        Human description.

    Examples
    --------
    .. code-block:: python

        >>> import brainunit as u
        >>> from brainpy_state._nest._validation.tolerance_conventions import CAT_B
        >>> u.get_unit(CAT_B.atol) == u.mV
        True
        >>> CAT_B.rtol
        1e-06
    """
    atol: object
    rtol: float
    align_steps: int = 0
    label: str = ""
    note: str = ""


@dataclasses.dataclass(frozen=True)
class DistributionalTolerance:
    """Multi-seed statistical tolerance (category D).

    Parameters
    ----------
    rate_rtol : float
        Relative tolerance on the seed-aggregated mean (network firing-rate parity).
    mean_diff_pct : float
        Tighter per-cell mean tolerance (convention for single-neuron distributional
        tests; recorded here for downstream clusters).
    autocorr_max_diff : float
        Absolute tolerance on the autocorrelation (convention; recorded for future use).
    n_seeds : int
        Minimum number of seeds per side.
    label : str, optional
        Category label (``"D"``).
    note : str, optional
        Human description.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy_state._nest._validation.tolerance_conventions import CAT_D
        >>> CAT_D.rate_rtol, CAT_D.n_seeds >= 4
        (0.05, True)
    """
    rate_rtol: float
    mean_diff_pct: float
    autocorr_max_diff: float
    n_seeds: int
    label: str = "D"
    note: str = ""


@dataclasses.dataclass(frozen=True)
class SpikeTimeTolerance:
    """Event-count / spike-time tolerance (category E).

    Parameters
    ----------
    max_count_diff : int
        Maximum allowed absolute difference in event count.
    max_peak_step_diff : int
        Maximum allowed absolute difference in peak step index (``1`` step ~ ``dt``).
    label : str, optional
        Category label (``"E"``).
    note : str, optional
        Human description.

    Examples
    --------
    .. code-block:: python

        >>> from brainpy_state._nest._validation.tolerance_conventions import CAT_E
        >>> CAT_E.max_count_diff, CAT_E.max_peak_step_diff
        (2, 1)
    """
    max_count_diff: int
    max_peak_step_diff: int
    label: str = "E"
    note: str = ""


#: Default simulation horizon (numerical-validation-gap.md section 6).
T_DEFAULT = 1000.0 * u.ms
#: Default integration step (numerical-validation-gap.md section 6).
DT_DEFAULT = 0.1 * u.ms
#: Default seed count for distributional (category D) comparisons.
N_SEEDS_DEFAULT = 5

#: A — adaptive numerical-integrator trace (RKF45: aeif / HH / izhikevich V_m).
CAT_A = TraceTolerance(1e-3 * u.mV, 1e-3, label="A",
                       note="adaptive RKF45 integrator V_m/state trace")
#: B — analytic exact-propagator trace (linear iaf_psc V_m / PSC), near-exact.
CAT_B = TraceTolerance(1e-6 * u.mV, 1e-6, label="B",
                       note="analytic exact-propagator trace (near-exact)")
#: B with recorder alignment — analytic family, allow a one-step multimeter offset.
CAT_B_ALIGNED = TraceTolerance(5e-2 * u.mV, 1e-3, align_steps=1, label="B",
                               note="analytic trace with one-step recorder alignment")
#: C — conductance / coupled deterministic trace (iaf_cond, multicompartment).
CAT_C = TraceTolerance(1e-3 * u.mV, 1e-3, label="C",
                       note="conductance/coupled deterministic trace")
#: C (rate) — mean-field rate fixed point (Hz, deterministic, pure-relative).
CAT_C_RATE = TraceTolerance(0.0, 5e-2, label="C",
                            note="mean-field rate fixed point (Hz), deterministic")
#: D — distributional / PRNG-divergent multi-seed statistic.
CAT_D = DistributionalTolerance(5e-2, 2e-2, 5e-2, N_SEEDS_DEFAULT,
                                note="PRNG-divergent multi-seed statistic; compare aggregate, never per-sample")
#: E — spike-time / event-count (precise spiking, PSC peak timing, event counts).
CAT_E = SpikeTimeTolerance(2, 1, note="spike-time/event-count; +/-1 step ~ dt")
