# K0 T8: voltage-gated conductance ports and NMDA

_Date: 2026-10-22_ · Parent: [`2026-10-21-k0-kernel-spec.md`](2026-10-21-k0-kernel-spec.md)
§2 ("NMDA conductance with Mg block -- voltage-gated conductance port, forward only"),
§4.2 (`Conductance(unit, E_rev, gate)`), §5 row T8. Architecture doc §7 (ports, single
owner of kinetics).

## Goal

Deliver a conductance whose effect is gated by the membrane potential, `I = g B(V) (E - V)`,
with `B(V)` evaluated **inside `derivatives` at every RK substep**, and an NMDA receptor
(saturating second-order kinetics) that feeds such a port. Fix the goal-01 bugs by
construction: Mg-block constants carry units (mM, 1/mV, mV), the NMDA open fraction is
dimensionless and the conductance scale `g_max` is explicit (the projection weight, nS).

## Modules

| module | contents |
|---|---|
| `_kernel/gates.py` | `MgBlock(Mg=1.2 mM, alpha=0.062/mV, beta=3.57 mM, V_offset=0 mV)`: `B(V) = 1 / (1 + Mg/beta exp(-alpha (V - V_offset)))` (Jahr & Stevens 1990). Constants are stored as canonical floats (hashable, precision-neutral); `V` must carry a voltage unit. |
| `_kernel/model.py` | generic mechanism: `NeuronModel.conductance_port_current(inp, V, p)` sums `conductance_current` over the **unfiltered** `Conductance` input ports; `with_conductance_ports(Model, name=None, params=None, **ports)` derives a model class with extra unfiltered (optionally gated) conductance ports. Models opt in with `accepts_conductance_ports = True` and call the helper in `derivatives`. |
| `_kernel/models/iaf_cond_exp.py`, `aeif_cond_alpha.py` | opt in (one added term in `dV/dt`; zero when no extra port is declared, so existing behaviour is bit-identical). |
| `_kernel/models/nmda.py` | `nmda_kinetics(x, s, pulse, arrived, p, dt)` (pure step) and the `NMDA` receptor population (pre-aligned kinetics owner). |
| `_kernel/network.py` | the T5 `NotImplementedError` for gated conductance ports is removed. |

## NMDA kinetics (from `_brainpy/synapse.py` `BioNMDA`)

```
dx/dt = alpha2 T(t) (1 - x) - beta2 x          (x: dimensionless)
ds/dt = alpha1 x (1 - s) - beta1 s             (s: open fraction, dimensionless, in [0, 1])
```

`T(t) = T` (mM) for `T_dur` after a presynaptic spike arrival, else 0. One step: an
arrival (payload > 0, multiplicity ignored -- the pulse saturates) restarts the pulse
counter at `grid_steps(T_dur, dt)` steps; the pulse is active this step if the counter is
positive; `x` is advanced exactly with `T` constant over the step, then `s` exactly with
the new `x` constant over the step (identical to `BioNMDA`'s two `exp_euler_step`s on
these linear ODEs); the counter is decremented. Saturation is `1 - x`, `1 - s` on
dimensionless states: it cannot depend on the conductance unit (goal-01 bug).

`NMDA(n)`: input `spike` `Event` (dimensionless weight), output `s` `Graded(1)`, states
`x`, `s`, `pulse` (int32). It is a *population of presynaptic receptor pools*: connect
`pre -> NMDA` `one_to_one` (default delay one step; a delay of 0 reproduces `BioNMDA`'s
same-step onset), then `NMDA -> post` on a gated conductance port with **weight =
g_max in a conductance unit**. The post neuron receives `g = sum_j w_ij s_j` once per
step (constant over the substeps); `B(V)` and `E - V` are re-evaluated at every substep.

Folding into T6: the kinetics are a pre-aligned (`alignment='pre'`), nonlinear
`SynapseModel` whose output kind is a dimensionless conductance fraction scaled by the
weight; it must be excluded from the post-merge pass. `nmda_kinetics` is the step
function to reuse; the `NMDA` population becomes a thin compatibility wrapper.

## Generic gated port

```python
NMDACell = with_conductance_ports(
    iaf_cond_exp, nmda=Conductance(u.nS, E_rev='E_nmda', gate=MgBlock()),
    params=dict(E_nmda=0.0 * u.mV))
```

Validation (class creation): base must opt in; new ports must be `Conductance`,
unfiltered, with `E_rev` (a Quantity or a parameter name declared by the base or `params`);
names must not clash. Event sources cannot feed an unfiltered conductance port directly
(kind check: a synapse model or a graded source is required).

## Tests (`gates_test.py`, `models/nmda_test.py`, `gated_conductance_test.py`)

- MgBlock vs analytic Jahr-Stevens with units (mV and V inputs give the same curve);
  unitless `V` raises; unit validation of constants; `Mg = 0` gives `B = 1`.
- Gate per substep: kernel vs independent RK4 reference with the gate live (agree to
  round-off) and with the gate frozen at the step start (differ); the live scheme
  converges with substeps (4th order), the frozen one does not (O(dt) plateau).
- NMDA open fraction vs `BioNMDA` (unitless-g configuration) and the full PSP vs a
  reference built from `BioNMDA` + `MgBlock` (unitless) integrated with RK4.
- Unit invariance (weights nS vs uS, port declared in uS, rates in 1/s); float32 vs
  float64; `jit`, `vmap` (batched runner), `for_loop`; network end to end.
- Optional Brian2 comparison of a single neuron driven by the NMDA conductance
  (skipped when `brian2` is absent).
