# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
r"""NEST-faithful ``stdp_facetshw_synapse_hom`` — FACETS/BrainScaleS hardware STDP.

Rebuilt as a frozen parameter spec plus a pure, vectorized
``update(state, ctx) -> (new_state, w_eff)`` rule kernel on
:class:`~brainpy_state._nest_network.event_plastic.EventPlasticProj`. Unlike the pair-based
``stdp_*`` models this is a *hardware* model (Schemmel et al. 2006; Pfeil et al. 2012):
the synapse holds a 4-bit discrete weight and two analogue charges, and a periodic
controller (the "readout cycle") quantises the weight, compares the charges to
thresholds through two configurable evaluation functions, applies one of three
look-up tables, and resets the charges (``stdp_facetshw_synapse_hom.h`` ``send()``).

Two charges accumulate between readouts, both in the substrate's ``'nearest'`` mode:

* ``a_causal`` — the *first* post since the last pre, paired with that pre
  (``exp(-(t_post - t_pre)/tau_plus)``); potentiation evidence;
* ``a_acausal`` — the *nearest* post before this pre, paired with this pre
  (``exp(-(t_pre - t_post)/tau_minus)``); depression evidence.

Both are folded in at the **pre** step, because NEST runs the readout *before*
accumulating the current ``send``'s pairing — so the causal contribution captured at a
post is deferred (``causal_pending``) and folded at the next pre, after that pre's
readout.
"""
from __future__ import annotations
from brainpy_state._nest_base.base import NESTPlasticity

import jax.numpy as jnp
import numpy as np
import brainunit as u
from brainstate.typing import ArrayLike

from brainpy_state._nest_base.plastic_base import (
    to_ms, to_scalar_float, to_scalar_int, unit_of,
    validate_delay, validate_receptor_type, weight_to_pa,
)

__all__ = ['stdp_facetshw_synapse_hom']

# NEST common-properties defaults (stdp_facetshw_synapse_hom_impl.h)
_DEFAULT_LUT_0 = (2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 15)
_DEFAULT_LUT_1 = (0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13)
_DEFAULT_LUT_2 = tuple(range(16))
_DEFAULT_CONFIG_0 = (0, 0, 1, 0)
_DEFAULT_CONFIG_1 = (0, 1, 0, 0)
_DEFAULT_RESET_PATTERN = (1, 1, 1, 1, 1, 1)


class stdp_facetshw_synapse_hom(NESTPlasticity):
    r"""FACETS/BrainScaleS hardware STDP synapse spec (NEST ``stdp_facetshw_synapse_hom``).

    A hardware-constrained STDP model with a **4-bit discrete weight** updated only at
    periodic *readout* events, not at every spike. Between readouts two analogue charges
    accumulate from nearest-neighbour pre/post pairings:

    .. math::

       a_\text{causal} \mathrel{+}= \exp\!\Big(\tfrac{-(t_\text{post}-t_\text{pre})}{\tau_+}\Big)
       \quad\text{(first post since the last pre)}

       a_\text{acausal} \mathrel{+}= \exp\!\Big(\tfrac{-(t_\text{pre}-t_\text{post})}{\tau_-}\Big)
       \quad\text{(nearest post before this pre)}

    At a readout (the first pre spike that crosses ``next_readout``) the weight is
    quantised to a LUT index ``e = round(w / w_\text{ple})``, two boolean evaluation
    functions compare the charges against thresholds, and one look-up table maps ``e``:

    .. math::

       \text{eval}_k = \frac{a_{tl} + b^k_2 a_\text{causal} + b^k_1 a_\text{acausal}}
                            {1 + b^k_2 + b^k_1}
                     > \frac{a_{th} + b^k_0 a_\text{causal} + b^k_3 a_\text{acausal}}
                            {1 + b^k_0 + b^k_3}

    with config bits :math:`b^k = `\ ``configbit_k``. ``(eval0, eval1)`` selects the
    table: ``(T,F)``\ ->\ ``lookuptable_0`` (potentiation), ``(F,T)``\ ->\ ``lookuptable_1``
    (depression), ``(T,T)``\ ->\ ``lookuptable_2`` (default identity), ``(F,F)`` leaves
    the index unchanged. The new weight is ``e' * w_\text{ple}`` — so the weight is
    **re-quantised at every readout even when no table fires**. ``reset_pattern`` then
    zeroes the charges (per branch, causal/acausal), and ``next_readout`` advances by
    whole ``readout_cycle_duration``\ s past the spike. With the default config bits and
    ``a_thresh_th == a_thresh_tl``, ``eval0`` reduces to ``a_causal > a_thresh_th`` and
    ``eval1`` to ``a_acausal > a_thresh_th``.

    .. warning::

       **Default-weight footgun.** With ``Wmax=100`` the quantum is
       ``w_ple = Wmax/15 ≈ 6.667``; the *default* ``weight=1.0`` quantises to index
       ``round(1/6.667)=0`` and the first readout zeroes it. Choose a weight on (or near)
       the LUT grid — e.g. ``5 * Wmax/15 ≈ 33.33``.

    Parameters
    ----------
    weight : ArrayLike or Quantity, optional
        Per-edge weight (pA; bare numbers are pA). Default ``1.0`` pA — see the footgun
        warning above.
    delay : Quantity, optional
        Homogeneous axonal delay (> 0). Default ``1.0 ms``.
    receptor_type : int, optional
        Postsynaptic receptor port (>= 0). Default ``0``.
    tau_plus : Quantity, optional
        Causal-charge (``K+``) trace constant (> 0). Default ``20.0 ms``.
    tau_minus : Quantity, optional
        Acausal-charge (``K-``) trace constant (> 0). Default ``20.0 ms``.
    Wmax : float, optional
        Weight bound, used for the default quantum. Default ``100.0``.
    a_thresh_th, a_thresh_tl : float, optional
        Upper/lower charge thresholds in the evaluation functions. Default ``21.835``.
    lookuptable_0, lookuptable_1, lookuptable_2 : sequence of int, optional
        The 16-entry weight-update tables (potentiation / depression / identity). Each
        entry must index back into the table (``[0, 15]``).
    configbit_0, configbit_1 : sequence of int, optional
        Four 0/1 bits parameterising ``eval0`` / ``eval1``.
    reset_pattern : sequence of int, optional
        Six 0/1 flags: reset (causal, acausal) after a ``lookuptable_0`` / ``_1`` / ``_2``
        update respectively. Default all-on.
    weight_per_lut_entry : float, optional
        The weight quantum. Default ``Wmax / (lut_size - 1)``.
    synapses_per_driver : int, optional
        Synapses served by one hardware weight-update controller. Default ``50``.
    driver_readout_time : float, optional
        Time (ms) one controller takes per synapse. Default ``15.0``.

    Notes
    -----
    **Single-driver scope.** For a single edge (and any edge count up to
    ``synapses_per_driver``) NEST's
    ``readout_cycle_duration = int((no_synapses-1)/synapses_per_driver + 1) *
    driver_readout_time`` collapses to ``driver_readout_time`` and every synapse's first
    readout offset is 0, which is what this spec models (``next_readout`` starts at 0).
    True multi-driver round-robin grouping for ``E > synapses_per_driver`` (staggered
    per-synapse offsets) is out of scope. See ``develop/NEST_PARITY_LEDGER.md`` Lessons (05).

    **Deferred accumulation.** NEST accumulates the ``send``'s pairing *after* the
    readout, so a readout never sees the charge from the pair that triggered it. The
    kernel reproduces this by capturing the causal term at the post (``causal_pending``)
    and folding it — together with the acausal term — only at the next pre, after that
    pre's readout. Exact pre/post coincidence follows the substrate second-latest
    convention and is not asserted against NEST.

    **Parity note.** The charge-accumulation / LUT-readout pairing convention, the
    NEST keys and single-driver scope, and the parity test are documented in
    :doc:`/nest-guide/stdp-divergences` (:ref:`stdp-facetshw`).

    References
    ----------
    .. [1] NEST ``models/stdp_facetshw_synapse_hom.h`` / ``_impl.h`` (``send()``: readout
       + LUT at the controller boundary, charge accumulation in ``get_history``). Schemmel,
       Gruebl, Meier & Mueller (2006) IJCNN; Pfeil et al. (2012) Front. Neurosci. 6:90.

    See Also
    --------
    stdp_synapse, stdp_nn_symm_synapse, stdp_nn_restr_synapse

    Examples
    --------
    .. code-block:: python

       >>> import brainunit as u
       >>> from brainpy.state import stdp_facetshw_synapse_hom
       >>> s = stdp_facetshw_synapse_hom(weight=33.333, Wmax=100.0)
       >>> round(s.weight_per_lut_entry, 3)
       6.667
       >>> s._weight_to_entry(33.333)        # nearest 4-bit index
       5
       >>> sorted(s.edge_state_init())
       ['a_acausal', 'a_causal', 'causal_pending', 'next_readout', 'post_seen', 'pre_seen']
    """
    __module__ = 'brainpy.state'

    is_homogeneous_weight = False
    stochastic = False
    # both charges read the substrate's per-neuron nearest trace
    pre_trace_mode = 'nearest'
    post_trace_mode = 'nearest'

    def __init__(
        self,
        weight: ArrayLike = 1.0,
        delay: ArrayLike = 1.0 * u.ms,
        receptor_type: int = 0,
        tau_plus: ArrayLike = 20.0 * u.ms,
        tau_minus: ArrayLike = 20.0 * u.ms,
        Wmax: ArrayLike = 100.0,
        a_thresh_th: ArrayLike = 21.835,
        a_thresh_tl: ArrayLike = 21.835,
        lookuptable_0=_DEFAULT_LUT_0,
        lookuptable_1=_DEFAULT_LUT_1,
        lookuptable_2=_DEFAULT_LUT_2,
        configbit_0=_DEFAULT_CONFIG_0,
        configbit_1=_DEFAULT_CONFIG_1,
        reset_pattern=_DEFAULT_RESET_PATTERN,
        weight_per_lut_entry: ArrayLike = None,
        synapses_per_driver: ArrayLike = 50,
        driver_readout_time: ArrayLike = 15.0,
    ):
        super().__init__(in_size=1)
        self.weight = weight_to_pa(weight)
        self.weight_unit = unit_of(self.weight)
        validate_delay(delay)
        self.delay = delay
        self.receptor_type = validate_receptor_type(receptor_type)

        self._tau_plus_ms = to_ms(tau_plus, name='tau_plus')
        self.tau_plus = self._tau_plus_ms * u.ms
        self._tau_minus_ms = to_ms(tau_minus, name='tau_minus')
        self.tau_minus = self._tau_minus_ms * u.ms
        if self._tau_plus_ms <= 0.0:
            raise ValueError("'tau_plus' must be > 0.")
        if self._tau_minus_ms <= 0.0:
            raise ValueError("'tau_minus' must be > 0.")

        self.Wmax = to_scalar_float(Wmax, name='Wmax')
        self.a_thresh_th = to_scalar_float(a_thresh_th, name='a_thresh_th')
        self.a_thresh_tl = to_scalar_float(a_thresh_tl, name='a_thresh_tl')

        # look-up tables (equal length; entries index back into the table)
        self.lookuptable_0 = self._validate_lut(lookuptable_0, name='lookuptable_0')
        self.lookuptable_1 = self._validate_lut(lookuptable_1, name='lookuptable_1')
        self.lookuptable_2 = self._validate_lut(lookuptable_2, name='lookuptable_2')
        self._lut_size = len(self.lookuptable_0)
        if not (len(self.lookuptable_1) == len(self.lookuptable_2) == self._lut_size):
            raise ValueError('look-up tables must have equal length.')
        self.configbit_0 = self._validate_configbit(configbit_0, name='configbit_0')
        self.configbit_1 = self._validate_configbit(configbit_1, name='configbit_1')
        self.reset_pattern = self._validate_reset(reset_pattern)

        self.synapses_per_driver = to_scalar_int(synapses_per_driver, name='synapses_per_driver')
        self.driver_readout_time = to_scalar_float(driver_readout_time, name='driver_readout_time')

        # weight quantum: defaults to Wmax/(lut_size-1) (NEST common-properties ctor)
        if weight_per_lut_entry is None:
            self.weight_per_lut_entry = self.Wmax / (self._lut_size - 1)
        else:
            self.weight_per_lut_entry = to_scalar_float(weight_per_lut_entry,
                                                        name='weight_per_lut_entry')
        # single-driver readout cadence (E <= synapses_per_driver): == driver_readout_time
        self.readout_cycle_duration = self.driver_readout_time

        # substrate per-neuron nearest traces drive both charges
        self.pre_trace_tau = self.tau_plus
        self.post_trace_tau = self.tau_minus

        # precomputed unit-free kernel constants
        self._wple = float(self.weight_per_lut_entry)
        self._readout_dur = float(self.readout_cycle_duration)
        self._lut0 = jnp.asarray(self.lookuptable_0, dtype=jnp.int32)
        self._lut1 = jnp.asarray(self.lookuptable_1, dtype=jnp.int32)
        self._lut2 = jnp.asarray(self.lookuptable_2, dtype=jnp.int32)
        self._cb0 = self.configbit_0
        self._cb1 = self.configbit_1
        self._rp = self.reset_pattern

    # -- validation helpers ------------------------------------------------
    @staticmethod
    def _validate_lut(table, *, name):
        t = tuple(int(x) for x in table)
        n = len(t)
        if n == 0:
            raise ValueError(f"'{name}' must be non-empty.")
        if any(e < 0 or e >= n for e in t):
            raise ValueError(f"'{name}' entries must be in [0, {n - 1}].")
        return t

    @staticmethod
    def _validate_configbit(bits, *, name):
        b = tuple(int(x) for x in bits)
        if len(b) != 4:
            raise ValueError(f"'{name}' must have 4 entries.")
        if any(x not in (0, 1) for x in b):
            raise ValueError(f"'{name}' entries must be 0 or 1.")
        return b

    @staticmethod
    def _validate_reset(pattern):
        p = tuple(bool(int(x)) for x in pattern)
        if len(p) != 6:
            raise ValueError("'reset_pattern' must have 6 entries.")
        return p

    def edge_state_init(self) -> dict:
        # two analogue charges, the deferred causal term, two eligibility flags, and the
        # per-edge readout clock (starts at 0 -> first pre triggers a no-op readout)
        return {'a_causal': 0.0, 'a_acausal': 0.0, 'causal_pending': 0.0,
                'pre_seen': 0.0, 'post_seen': 0.0, 'next_readout': 0.0}

    # -- quantisation helpers (host-side, for tests / introspection) -------
    def _weight_to_entry(self, w) -> int:
        """Quantise a weight to its 4-bit LUT index (NEST ``weight_to_entry_``)."""
        e = int(np.floor(float(w) / self._wple + 0.5))         # round-half-up (w >= 0)
        return int(np.clip(e, 0, self._lut_size - 1))

    def _entry_to_weight(self, entry) -> float:
        """Map a LUT index back to a weight (NEST ``entry_to_weight_``)."""
        return float(entry) * self._wple

    def _eval(self, a_causal, a_acausal, cb):
        """The hardware evaluation function for one set of config bits (NEST ``eval_function_``)."""
        b0, b1, b2, b3 = cb
        lhs = (self.a_thresh_tl + b2 * a_causal + b1 * a_acausal) / (1.0 + b2 + b1)
        rhs = (self.a_thresh_th + b0 * a_causal + b3 * a_acausal) / (1.0 + b0 + b3)
        return lhs > rhs

    # -- rule kernel -------------------------------------------------------
    def update(self, state, ctx):
        w = state['weight']
        a_causal = state['a_causal']
        a_acausal = state['a_acausal']
        causal_pending = state['causal_pending']
        pre_seen = state['pre_seen']
        post_seen = state['post_seen']
        next_readout = state['next_readout']

        pre_fired = ctx.pre_spike > 0
        post_fired = ctx.post_spike > 0
        # nearest partner trace excluding this step's own spike (second-latest on coincide)
        kplus = ctx.pre_trace - ctx.pre_spike                  # exp(-(t - t_lastpre)/tau_plus)
        kminus = ctx.post_trace - ctx.post_spike               # exp(-(t - t_lastpost)/tau_minus)

        # === 1. READOUT — a pre that crosses next_readout, using the CURRENT charges ===
        do_readout = pre_fired & (ctx.t_now > next_readout)
        dw = jnp.clip(jnp.floor(w / self._wple + 0.5), 0.0, self._lut_size - 1).astype(jnp.int32)
        eval0 = self._eval(a_causal, a_acausal, self._cb0)
        eval1 = self._eval(a_causal, a_acausal, self._cb1)
        sel_tf = eval0 & (~eval1)                              # potentiation -> LUT0
        sel_ft = (~eval0) & eval1                              # depression  -> LUT1
        sel_tt = eval0 & eval1                                 # both        -> LUT2
        new_dw = jnp.where(sel_tf, self._lut0[dw],
                           jnp.where(sel_ft, self._lut1[dw],
                                     jnp.where(sel_tt, self._lut2[dw], dw)))
        w_readout = new_dw.astype(w.dtype) * self._wple        # re-quantised even on (F,F)
        reset_c = (sel_tf & self._rp[0]) | (sel_ft & self._rp[2]) | (sel_tt & self._rp[4])
        reset_ac = (sel_tf & self._rp[1]) | (sel_ft & self._rp[3]) | (sel_tt & self._rp[5])
        # advance the readout clock past t by whole cycles (NEST while-loop, closed form)
        n_adv = jnp.floor((ctx.t_now - next_readout) / self._readout_dur) + 1.0
        next_readout_adv = next_readout + n_adv * self._readout_dur

        w = jnp.where(do_readout, w_readout, w)
        a_causal = jnp.where(do_readout & reset_c, 0.0, a_causal)
        a_acausal = jnp.where(do_readout & reset_ac, 0.0, a_acausal)
        next_readout = jnp.where(do_readout, next_readout_adv, next_readout)

        # === 2. capture the causal term at the FIRST post since the last pre ===
        capture = post_fired & (pre_seen > 0)
        causal_pending = jnp.where(capture, kplus, causal_pending)

        # === 3. fold deferred charges at the PRE step (NEST accumulates after readout) ===
        a_causal = a_causal + jnp.where(pre_fired, causal_pending, 0.0)
        a_acausal = a_acausal + jnp.where(pre_fired & (post_seen > 0), kminus, 0.0)
        causal_pending = jnp.where(pre_fired, 0.0, causal_pending)

        # === 4. eligibility flags: a spike makes its side seen, clears the opposite ===
        new_pre_seen = jnp.where(pre_fired, 1.0, jnp.where(post_fired, 0.0, pre_seen))
        new_post_seen = jnp.where(post_fired, 1.0, jnp.where(pre_fired, 0.0, post_seen))

        return {'weight': w, 'a_causal': a_causal, 'a_acausal': a_acausal,
                'causal_pending': causal_pending, 'pre_seen': new_pre_seen,
                'post_seen': new_post_seen, 'next_readout': next_readout}, w
