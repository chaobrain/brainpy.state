# K0 T6 — `SynapseModel` (Expon + merge pass) and `PlasticityRule` (static / stdp, lazy timing)

_Date: 2026-10-22_ · Parent: [`2026-10-21-k0-kernel-spec.md`](2026-10-21-k0-kernel-spec.md)
§4.2 (single owner), §4.6, §5 row T6, §6 G8 · Architecture doc §7 (`SynapseModel`,
`PlasticityRule`, §7.1 lazy timing), §8.2.

## Modules

| module | contents |
|---|---|
| `_kernel/synapse.py` | `SynapseModel` contract, `Expon`, `SynapseGroup`, `plan_synapse_groups` (merge pass) |
| `_kernel/plasticity.py` | `PlasticityRule` protocol, `PlasticContext`, `FastForm`, `static_synapse`, `stdp_synapse`, `PlasticPlan` (compiled per projection: both paths) |
| `network.py` / `compiler.py` / `runner.py` | `connect(..., synapse=, plasticity=)`, compile-time grouping, deliver/emit integration, `Plan.describe()` "Merges", `Runner.edges` |

## SynapseModel

A synapse model sits between edge delivery and the port deposit. Contract:

```python
class SynapseModel:
    alignment = 'post'            # K0: post-aligned only (state per target neuron)
    linear: bool                  # linear kinetics -> mergeable when params equal
    filters = True                # owns kinetics -> single-owner check vs PortKind.filtered
    def output_kind(self, dst) -> PortKind          # what the port receives (value kind)
    def init_state(self, n, dtype) -> dict          # mantissas, (n,)
    def step(self, state, x, ctx) -> (state, out)   # x: (n,) summed edge input, port units
    def merge_key(self) -> Hashable | None          # None = never merged
```

`Expon(tau, merge=True)`: `g <- g * exp(-dt/tau) + x`, output `g` (the old
`_brainpy/exponential.py` order: exponential-Euler decay — exact for this linear ODE —
then the delta input). Accepts **event** sources only, onto **unfiltered** `Current` or
`Conductance` ports (weights carry the port unit; conductance weights non-negative).
The output enters the port like any value input (a `Current` port latches it for the
next step, a `Conductance` port sees it inside the ODE).

**Single owner.** `connect(..., synapse=Expon(...))` onto a port whose kind is
`filtered=True` (the neuron already filters, e.g. `iaf_cond_exp.ex`) raises `PortError`.
An event source onto an unfiltered `Current`/`Conductance` port without a synapse is
still rejected by the T1 kind check.

**Merge pass** (compile, `net.compile(..., merge=True)` default). Synapse states are
grouped per `(target population, port, merge_key)`; a group with several projections
holds **one** `(n_pop,)` state fed by the sum of their deliveries (the AlignPost
`O(N_post)` result). Non-linear models return `merge_key() is None`; `merge=False`
(compile) or `Expon(merge=False)` opts out. `Plan.describe()` lists the merged groups
under "Merges". Merged and unmerged runs are mathematically identical (linearity); in
floating point they agree to rounding, and bit-for-bit when the decay factor and the
weights are dyadic (tested with `tau = dt / ln 2`).

## PlasticityRule

```python
class PlasticityRule:
    params: dict[str, float]           # rule parameters, traced (ConstParamState / ParamState)
    trainable: tuple[str, ...]         # which params are ParamState (learned plasticity)
    traces: dict[str, str]             # {'pre'|'post': <tau parameter name>}; all-to-all
    reads: tuple[str, ...] = ()        # post-state reads (none in K0)
    post_port = 'spike'                # post output port feeding the post trace
    axonal_delay: Quantity = 0 ms      # dendritic = total delay - axonal
    timing: 'lazy' | 'eager'
    def edge_state_init(self) -> dict                  # extra per-edge state {name: init}
    def update(self, state, ctx) -> (state, w_eff)     # pure jnp over (E,) arrays
    def fast_form(self) -> FastForm | None             # "add trace, clip" form
```

`PlasticContext(pre_spike, post_spike, pre_trace, post_trace, params, t, dt, key)`:
per-edge `(E,)` arrays in CSR order; `pre_spike` / `post_spike` are the events **at the
synapse** this step; traces are the **strictly prior** values (NEST `get_K_value`:
the decayed trace before this step's own event is added — the old `trace - spike`
exclusion without its rounding).

### Timing (NEST lazy, delay split)

NEST `stdp_synapse` treats the whole delay `d` as dendritic. In general, with
`d = d_ax + d_den` (steps, `d_den >= 1`):

* pre spike of step `n` reaches the synapse at step `n + d_ax`;
* post spike of step `m` reaches the synapse at step `m + d_den` (read from a
  pre-side ring on the post population's spike port — the dendritic seam);
* at synapse step `s` (emit phase, after all populations updated): decay traces,
  run the rule on post arrivals (facilitation, using `K+`) then on pre arrivals
  (depression, using `K-`), add the step's events to the traces;
* the delivered amplitude `y = W_eff^T x` is deposited in a post-side arrival ring
  `d_den` steps ahead, so it acts at `n + d_ax + d_den = n + d` with the weight of
  **its own** send (later sends cannot overwrite in-flight events).

This reproduces NEST's `send()` exactly: post spikes with `t_last - d < t_q <= t_pre - d`
facilitate in time order before the depression read `K-(t_pre - d)` (strictly prior).
**Lazy** (default): the stored (observable) weight changes only on pre sends —
`committed = where(pre_at_synapse, working, committed)`, an extra `(E,)` array;
**eager**: the working weight is the observable (no extra array). Delivered
amplitudes are identical in both modes.

### Two paths

* **fast** — `brainevent.update_csr_on_binary_post` (facilitation) then
  `update_csr_on_binary_pre` (depression) over the CSR/CSC structure. Eligible when
  the rule's `fast_form()` exists (stdp: `mu_plus == mu_minus == 0`, additive), no
  rule parameter is trainable, and the initial weights lie in the clip interval
  (brainevent clips *all* weights). Forward only (no VJP upstream).
* **reference** — the rule's `update` over gathered `(E,)` arrays; differentiable
  (gradients w.r.t. rule params through a rollout, e.g. `lambda`, `alpha`, `tau_plus`).

`plasticity_path='auto'|'fast'|'reference'` on the rule; `'auto'` picks fast when
eligible. Plastic weights are per-edge **CSR** (`storage='dense'` is rejected: the
dense round trip is lossy with multapses). K0 restrictions (clear errors): scalar
(homogeneous) delay, `trainable=True` weights on a plastic projection (hybrid L4 is
K1), event sources only.

`static_synapse()` compiles to an ordinary (non-plastic) projection.

## Gates and tests

* G8 *(NEST)*: `stdp_synapse` weight at every send vs NEST 3.10 `weight_recorder`,
  pair window sweep, coincident pre/post-arrival, a long train, dendritic delays
  1.0 and 2.5 ms, multiplicative (default) and additive; `CAT_B`; the kernel side
  is a real network (`spike_generator` pre, `LIF` post forced by a
  `spike_generator` driver) — no dendritic shift is faked.
* fast == reference (additive stdp, random network) to rounding.
* Expon PSC vs analytic, vs the old `_brainpy` `Expon`, PSP vs the closed form of the
  exact discrete recurrence; merged == unmerged; single-owner errors.
* gradient of final weights w.r.t. `lambda` / `alpha` through a rollout vs central
  finite differences.
