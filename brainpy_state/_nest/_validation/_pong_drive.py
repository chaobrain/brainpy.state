# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""Shared live-NEST drives for the §3.10 pong learners (R-STDP + dopaminergic).

The pong demos are reinforcement learners: their game-by-game weight trajectories
PRNG-diverge from NEST, so there is no per-sample full-game parity (see the cluster
note in the test modules). Parity is therefore split into **deterministic component
checks** that this module backs:

1. ``calculate_stdp`` host-equality — the R-STDP correlation is pure float math on
   motor spike *times*; we run NEST's *actual* ``PongNetRSTDP.calculate_stdp`` source
   (loaded from the NEST examples, called on a lightweight stub so no NEST kernel
   network is built) and assert our port reproduces it bit-for-bit.

The dopaminergic critic component drive (Stage E) is added below when that learner
lands. Everything here is import-light: the NEST example module is loaded lazily and
only inside ``@requires_nest`` tests.
"""
import importlib.util
import os
import types

import numpy as np

#: Location of NEST's ``pong/`` example sources (overridable for other checkouts).
NEST_PONG_DIR = os.environ.get(
    'NEST_PONG_DIR',
    '/mnt/d/codes/githubs/computational_neuroscience/nest-simulator/pynest/examples/pong',
)


def _load_nest_networks():
    """Import NEST's ``pong/networks.py`` as a module (lazily; needs live ``nest``).

    Loaded by explicit file path so it does not depend on the NEST examples being on
    ``sys.path``. Raises ``FileNotFoundError`` if the checkout is absent (the caller
    is an ``@requires_nest`` test, which also skips when ``import nest`` fails).
    """
    path = os.path.join(NEST_PONG_DIR, 'networks.py')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'NEST pong example not found at {path!r}; set NEST_PONG_DIR to the '
            "checkout's pynest/examples/pong directory.")
    spec = importlib.util.spec_from_file_location('nest_pong_networks', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # executes ``import nest`` at module top
    return mod


def nest_calculate_stdp(pre_spikes, post_spikes, *, only_causal=True, next_neighbor=True):
    """NEST ``PongNetRSTDP.calculate_stdp`` evaluated on a stub (no kernel network).

    Calls the unbound NEST method against a ``SimpleNamespace`` carrying only the three
    class constants it reads (``stdp_amplitude`` / ``stdp_tau`` / ``stdp_saturation``),
    so we exercise the genuine upstream source without constructing the spiking network
    its ``__init__`` would build.

    Parameters
    ----------
    pre_spikes, post_spikes : array_like
        Pre- / post-synaptic spike times (ms).
    only_causal, next_neighbor : bool, optional
        Forwarded to NEST's method. Defaults ``True``.

    Returns
    -------
    float
        NEST's accumulated STDP correlation.
    """
    cls = _load_nest_networks().PongNetRSTDP
    stub = types.SimpleNamespace(
        stdp_amplitude=cls.stdp_amplitude,
        stdp_tau=cls.stdp_tau,
        stdp_saturation=cls.stdp_saturation,
    )
    return cls.calculate_stdp(stub, np.asarray(pre_spikes, dtype=float),
                              np.asarray(post_spikes, dtype=float),
                              only_causal=only_causal, next_neighbor=next_neighbor)
