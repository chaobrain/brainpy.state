# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Scaled microcircuit check against NEST (the neuron's parity: ``models/nest_parity_test.py``).

Network: the microcircuit at ``scale = 0.02`` (both sides, 4 seeds) gives every population a rate
within the G2 rule's 5 % or 2 SEM bound or 1 Hz (small-network smoke test; the gate
itself is full scale, ``benchmarks/k1a_microcircuit.py``).
"""
from __future__ import annotations

import numpy as np

from brainpy_state._nest_validation.nest_compare import requires_nest

SMALL_SCALE = 0.02
SMALL_SEEDS = (1, 2, 3, 4)


@requires_nest
def test_scaled_microcircuit_rates_match_nest():
    """Scale 0.02 (1543 neurons, in-degrees kept), 4 seeds each side, 500 ms after 100 ms.

    Per population: the G2 rule, or ``|Δ| <= 1 Hz`` (the smallest populations have
    ~20 neurons here, so the seed spread alone is not a usable bound).
    """
    from brainpy_state._kernel.validation.microcircuit import (
        POPULATIONS,
        kernel_microcircuit,
        nest_microcircuit,
    )
    from brainpy_state._kernel.validation.stats import g2_rule

    ref = [nest_microcircuit(s, 100.0, 500.0, scale=SMALL_SCALE, threads=4)['rates']
           for s in SMALL_SEEDS]
    ours = [kernel_microcircuit(s, 100.0, 500.0, scale=SMALL_SCALE)['rates'] for s in SMALL_SEEDS]
    bad = []
    for k in POPULATIONS:
        r = g2_rule([o[k] for o in ours], [x[k] for x in ref], k)
        print(k, round(r.ours_mean, 3), round(r.ref_mean, 3), round(r.bound, 3))
        if not (r.passed or abs(r.diff) <= 1.0):
            bad.append(r)
    assert not bad, bad
    assert all(np.mean([x[k] for x in ref]) > 0.1 for k in POPULATIONS)  # network is active
