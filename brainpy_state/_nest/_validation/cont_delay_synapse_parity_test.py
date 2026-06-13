# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity: ``cont_delay_synapse`` (continuous, sub-dt delay).

``cont_delay_synapse`` delivers a fixed weight after a delay that need not be an
integer multiple of the step. Two regimes are validated against live NEST, weakest
assumption first so a sub-dt mismatch can never be masked by the grid case:

* **integer (grid) delay** — ``delay`` an exact multiple of ``dt`` -> NEST's
  ``delay_offset_`` is 0 and a grid ``iaf_psc_exp`` receives the full weight at
  ``delay/dt``, identical to ``static_synapse``. The post ``V_m`` trace matches
  NEST step-for-step (category B, one-step recorder alignment).

* **sub-dt delay** — ``delay`` off-grid (``frac = d/dt - floor != 0``). NEST's
  *precise* ``iaf_psc_exp_ps`` integrates the true off-grid arrival at ``t + d``,
  whereas the rebuilt substrate is a grid integrator and instead delivers at the
  integer floor and splits the amplitude across the two bracketing grid steps with
  a one-step FIR ``[1-frac, frac]`` (scheme a). That split conserves charge and
  places the arrival centroid exactly at ``d``, so the two faithful invariants —
  the **time-integrated depolarization** ``∫V_m`` and the **EPSP peak amplitude** —
  match the precise neuron to ``~1e-4`` relative, and the peak *timing* matches
  within one step (category E). Only the instantaneous sub-step *onset transient*
  differs (a bounded ripple at the PSC onset, ``max|ΔV_m| ~ 0.25 mV`` on a
  ``~4.3 mV`` EPSP here, i.e. ``frac``-dependent and first-order in ``dt/τ`` at the
  onset discontinuity, vanishing as ``dt -> 0``). A grid integrator *cannot* place
  an event between steps, so this residual is intrinsic, not a tolerance failure;
  it is asserted-around (integral + peak-step) and documented here, never hidden
  behind a loose per-sample bound.
"""
import unittest

import brainstate
import braintools
import jax
import jax.numpy as jnp
import numpy as np
import saiunit as u
from brainstate import transform

jax.config.update('jax_enable_x64', True)
brainstate.environ.set(precision=64, platform='cpu')

import brainpy_state
from brainpy_state import cont_delay_synapse
from brainpy_state._network import EventPlasticProj
from brainpy_state._nest._validation.nest_compare import compare_trace, requires_nest
from brainpy_state._nest._validation import tolerance_conventions as tc

try:
    import nest
    _HAS_NEST = True
except Exception:
    _HAS_NEST = False

# Post neuron: linear, never-spiking (V_th huge) iaf_psc_exp -> pure subthreshold.
_NPAR = dict(C_m=250., tau_m=20., tau_syn_ex=3.0, tau_syn_in=3.0,
             t_ref=2., E_L=0., V_reset=0., V_m=0., V_th=1e4)
_DT = 0.1
_W = 500.0
# Charge-exact invariant tolerance: the measured ∫V_m / peak-V_m relative error vs
# the precise neuron is ~1e-5..4e-5 across frac in {0.3, 0.5, 0.7}; 1e-3 keeps a
# >=25x margin while staying ~60x tighter than the instantaneous onset residual.
_CHARGE_RTOL = 1e-3


def _bp_post():
    return brainpy_state.iaf_psc_exp(
        1, C_m=_NPAR['C_m'] * u.pF, tau_m=_NPAR['tau_m'] * u.ms,
        tau_syn_ex=_NPAR['tau_syn_ex'] * u.ms, tau_syn_in=_NPAR['tau_syn_in'] * u.ms,
        t_ref=_NPAR['t_ref'] * u.ms, E_L=_NPAR['E_L'] * u.mV,
        V_reset=_NPAR['V_reset'] * u.mV, V_th=_NPAR['V_th'] * u.mV,
        V_initializer=braintools.init.Constant(_NPAR['V_m'] * u.mV))


def _bp_vm_trace(rule, spike_steps, n_steps):
    """Post V_m trace (mV) for a single 0->0 edge, driven inside ``for_loop``."""
    post = _bp_post()
    box = {'v': jnp.zeros(1)}
    proj = EventPlasticProj(
        pre_spike=lambda: box['v'], n_pre_pop=1, pre_local_idx=jnp.arange(1),
        post=post, post_local_idx=jnp.arange(1), n_post_pop=1,
        pre_idx=jnp.array([0]), post_idx=jnp.array([0]), rule=rule)
    brainstate.nn.init_all_states(post)
    brainstate.nn.init_all_states(proj)

    spikes = np.zeros((n_steps, 1))
    spikes[np.asarray(spike_steps, dtype=int), 0] = 1.0
    spikes = jnp.asarray(spikes)
    times = jnp.arange(n_steps) * _DT * u.ms
    indices = jnp.arange(n_steps)

    def step(t, i, x_in):
        box['v'] = x_in
        with brainstate.environ.context(t=t, i=i):
            proj.update()
            post.update()
            return u.get_mantissa(post.V.value[0] / u.mV)

    return np.asarray(transform.for_loop(step, times, indices, spikes))


@requires_nest
class TestContDelaySynapseParity(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def _nest_vm_trace(self, weight, delay, spike_times, T_ms, neuron="iaf_psc_exp"):
        nest.ResetKernel()
        nest.resolution = _DT
        n = nest.Create(neuron, 1, params=_NPAR)
        sg = nest.Create("spike_generator", params={"spike_times": list(spike_times)})
        mm = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": _DT})
        nest.Connect(sg, n, syn_spec={"synapse_model": "cont_delay_synapse",
                                      "weight": weight, "delay": float(delay)})
        nest.Connect(mm, n)
        nest.Simulate(T_ms)
        return np.asarray(mm.get("events")["V_m"])

    # -- layer 1: integer/grid delay is exact (verify first) --------------
    def test_grid_delay_matches_nest(self):
        # Integer-multiple delay -> offset 0 -> grid delivery == static_synapse.
        T_ms, delay = 60.0, 2.0                      # 20 steps, on grid
        nest_v = self._nest_vm_trace(_W, delay, [10.0], T_ms)
        bp_v = _bp_vm_trace(cont_delay_synapse(weight=_W * u.pA, delay=delay * u.ms),
                            [int(round(10.0 / _DT))], int(round(T_ms / _DT)))
        m = min(len(nest_v), len(bp_v))
        compare_trace(nest_v[:m], bp_v[:m], tol=tc.CAT_B_ALIGNED,
                      metric="cont_delay grid V_m").assert_()

    # -- layer 3: sub-dt vs the precise neuron (charge/centroid exact) -----
    def test_subdt_charge_exact_vs_precise(self):
        # vs NEST's precise iaf_psc_exp_ps (true off-grid arrival at t + d).
        T_ms = 120.0                                 # EPSP fully decays -> ∫ converges
        n_steps = int(round(T_ms / _DT))
        spike_steps = [int(round(10.0 / _DT))]
        for delay in (1.33, 1.35, 1.37):             # frac in {0.3, 0.5, 0.7}, k_lo = 13
            with self.subTest(delay=delay):
                nest_v = self._nest_vm_trace(_W, delay, [10.0], T_ms,
                                             neuron="iaf_psc_exp_ps")
                bp_v = _bp_vm_trace(
                    cont_delay_synapse(weight=_W * u.pA, delay=delay * u.ms),
                    spike_steps, n_steps)
                m = min(len(nest_v), len(bp_v))
                a, b = nest_v[:m], bp_v[:m]

                # (i) charge exact: time-integrated depolarization matches.
                int_ref, int_cand = float(a.sum()) * _DT, float(b.sum()) * _DT
                rel_int = abs(int_cand - int_ref) / abs(int_ref)
                self.assertLess(rel_int, _CHARGE_RTOL,
                                f"∫V_m rel {rel_int:.2e} >= {_CHARGE_RTOL:.0e} (d={delay})")

                # (ii) peak amplitude exact (broad PSC): matches the precise neuron.
                rel_peak = abs(float(b.max()) - float(a.max())) / abs(float(a.max()))
                self.assertLess(rel_peak, _CHARGE_RTOL,
                                f"peak V_m rel {rel_peak:.2e} (d={delay})")

                # (iii) peak timing within one step (category E); the only residual
                # is the sub-step onset transient, intrinsic to a grid integrator.
                dpeak = abs(int(np.argmax(b)) - int(np.argmax(a)))
                self.assertLessEqual(dpeak, tc.CAT_E.max_peak_step_diff,
                                     f"|Δpeak_step| {dpeak} > {tc.CAT_E.max_peak_step_diff}")


if __name__ == "__main__":
    unittest.main()
