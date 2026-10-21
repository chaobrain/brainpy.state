# docs-rebuild — shared conventions cheatsheet (READ FIRST)

This is the single source of truth every docs subagent must follow. It encodes
the **verified public API**, canonical code patterns, hard rules, and build
facts for the `brainpy.state` documentation rebuild. The full design is in
`2026-06-17-docs-rebuild-spec.md` (absolute:
`/mnt/d/codes/projects/brainpy-state/dev/superpowers/docs-rebuild-spec.md`).

---

## 0. Hard rules (NON-NEGOTIABLE)

1. **Namespace.** All example/doc code uses the **public** namespace only:
   `import brainpy` then `brainpy.state.X`, or `from brainpy import state as bs`.
   **NEVER** `import brainpy_state` and **never** touch `brainpy_state._*`
   private submodules in docs. (Direct import is blocked at runtime anyway.)
2. **Never drive a model with a bare Python `for`/`while` loop over time steps.**
   Use `brainstate.transform`:
   - one step / one-shot → `brainstate.transform.jit`
   - many steps, collect outputs → `brainstate.transform.for_loop`
   - many steps with explicit carry → `brainstate.transform.scan`
   - long rollout under autograd (BPTT) → `brainstate.transform.checkpointed_for_loop` / `checkpointed_scan`
   (An outer *optimization/epoch* loop that calls a `jit`-compiled train step is
   fine — the rule is about the time-stepping of the model.)
3. **No maturity labels.** Do NOT add "Stable"/"Experimental"/"Beta" badges,
   banners, or admonitions anywhere. Both families are presented as
   production-ready. Trust for NEST models is shown via the validation page.
4. **NumPy-style docstrings** for any Python API you write (rare in docs; mostly
   the package already has them). Section order: Summary, Extended, Parameters,
   Returns, Raises, See Also, Notes, References, Examples.
5. **Citations:** only **eLife BrainPy (2023)** + **ICLR 2024** in the
   cite-this-software block (`project/citing.rst`). The **2026 Nature Comms**
   paper appears ONLY as a related-work reference inside
   `concepts/online-learning.rst` — never in the citing block.
6. **Diagrams are placeholders.** The user generates images themselves with
   GPT-image-2. Where a diagram belongs, insert a `.. figure::` pointing to
   `/_static/concepts/diagram-N.png` with a caption + alt text, and a comment
   noting the image is user-supplied. Do NOT attempt to create images.
7. **Stay in your lane.** Only create/edit files in YOUR assigned paths. Do NOT
   touch `../index.rst`, `docs/conf.py`, or another part's directory. Do NOT
   run `git commit`/`git push`. Do NOT delete the old dirs (`brainpy-guide/`,
   `nest-guide/`, `nest-status/`, `quickstart/`, old `examples/gallery.rst`) —
   the orchestrator removes them after salvage.

---

## 1. Verified public API (from `dir(brainpy.state)`, 210 names)

**Base:** `Dynamics`, `Neuron`, `Synapse`, `Projection`, `SynOut`, `Network`,
`Builder`, `Simulator`, `SimulationResult`, `Recorder`, `NodeView`,
`SynapseCollection`.

**BrainPy-style neurons:** `IF`, `LIF`, `LIFRef`, `ALIF`, `ExpIF`, `ExpIFRef`,
`AdExIF`, `AdExIFRef`, `QuaIF`, `AdQuaIF`, `AdQuaIFRef`, `Gif`, `GifRef`,
`Izhikevich`, `IzhikevichRef`, `HH`, `WangBuzsakiHH`, `MorrisLecar`,
`FitzHughNagumo`, `HindmarshRose`, `CobaLIF`, `CubaLIF`.

**BrainPy-style synapses:** `Expon`, `DualExpon`, `Alpha`, `AMPA`, `GABAa`,
`BioNMDA`. (Note: it is `GABAa` and `BioNMDA` — there is no `GABA`/`NMDA`.)

**Synaptic outputs:** `COBA`, `CUBA`, `MgBlock`.

**Projections (classes):** `AlignPostProj`, `DeltaProj`, `CurrentProj`,
`AllToAllProj`, `OneToOneProj`, `FixedIndegreeProj`, `FixedOutdegreeProj`,
`FixedTotalNumberProj`, `PairwiseBernoulliProj`, `SymmetricPairwiseBernoulliProj`,
`PairwisePoissonProj`.
**Projections (functions):** `align_post_projection(...)`,
`align_pre_projection(...)`. **IMPORTANT:** AlignPre is the *function*
`align_pre_projection` — there is **no** `AlignPreProj` class. AlignPost has both
the `AlignPostProj` class and the `align_post_projection` function.

**Short-term plasticity:** `STP`, `STD`.

**Gap junctions:** `SymmetryGapJunction`, `AsymmetryGapJunction`.

**Readouts:** `LeakyRateReadout`.

**Inputs / encoders:** `ConstantInput`, `StepInput`, `RampInput`,
`SectionInput`, `SinusoidalInput`, `OUProcessInput`, `WienerProcessInput`,
`PoissonInput`, `PoissonEncoder`, `PoissonSpike`, `SpikeTime`.

**NEST-compatible base:** `NESTNeuron`, `NESTSynapse`, `NESTPlasticity`,
`NESTDevice`. **NEST models** are lowercase names like `iaf_psc_alpha`,
`iaf_psc_exp`, `aeif_cond_exp`, `aeif_psc_alpha`, `hh_psc_alpha`, `gif_psc_exp`,
`glif_psc`, `izhikevich`, `parrot_neuron`, rate models (`lin_rate_ipn`,
`tanh_rate_ipn`, `siegert_neuron`), devices (`poisson_generator`,
`spike_recorder`, `multimeter`, `voltmeter`, `dc_generator`,
`step_current_generator`, `spike_generator`, …), synapses (`static_synapse`,
`stdp_synapse`, `tsodyks_synapse`, …), `spatial`, `dist`, `network`.
(Get the authoritative list for a family by importing and filtering
`dir(brainpy.state)`, or read `docs/apis/nest-*.rst`.)

### Verified signatures (key ones)

```text
LIFRef(in_size, R=1*ohm, tau=…, V_rest=…, V_th=…, V_reset=…, tau_ref=…,
       V_initializer=…, spk_fun=…, …)
LIF(in_size, tau=…, V_rest=…, V_th=…, V_reset=…, spk_fun=…, …)
Expon(in_size, tau=8*ms, g_initializer=…)        # also Expon.desc(in_size, tau=…)
COBA(E)                                            # also COBA.desc(E=…)
CUBA(scale=Unit("V"))                              # also CUBA.desc(...)
AlignPostProj(*modules, comm: Callable, syn, out, post: Dynamics, label=None, delay=…)
DeltaProj(*prefetch, comm: Callable, post: Dynamics, label=None)
align_pre_projection(*spike_generator, syn: Dynamics, comm: Callable, out: SynOut, post: Dynamics, stp: Dynamics = None)
align_post_projection(*spike_generator, comm: Callable, syn, out, post: Dynamics, stp: Dynamics = None)
STP(in_size, U=0.15, tau_f=…, tau_d=…, …)
LeakyRateReadout(in_size, out_size, tau=5*ms, w_init=…)
```

The `syn=`/`out=` args of `AlignPostProj` take **describers**: use
`Expon.desc(...)` and `COBA.desc(...)` (not instances) — see canonical pattern.

### Connectivity (`brainstate.nn`)
`EventFixedProb(n_pre, n_post, conn_num=…, conn_weight=…)`, `EventFixedNumConn`,
`EventLinear`, `Linear`, `SignedWLinear`, `SparseLinear`, `ScaledWSLinear`,
`Sequential`.

### Transforms (`brainstate.transform`)
`jit`, `grad`, `for_loop`, `scan`, `cond`, `ifelse`,
`checkpointed_for_loop`, `checkpointed_scan`, `ProgressBar`, `eval_shape`,
`jacobian`, `hessian`, … .

### Tooling
- Units: `import brainunit as u` → `u.mV, u.ms, u.mS, u.mA, u.nS, u.Hz`,
  `u.math.arange/where/max/argmax/mean`, `u.Quantity`.
- Init: `braintools.init.Normal(mean, std, unit=u.mV)`, `KaimingNormal(scale=…, unit=u.mA)`,
  `ZeroInit(unit=…)`, `Constant(value)`.
- Surrogate: `braintools.surrogate.ReluGrad()` (and others) → pass as `spk_fun=`.
- Optim/metric/vis: `braintools.optim.Adam(lr=…)`, `braintools.metric.softmax_cross_entropy_with_integer_labels`, `braintools.visualize.get_figure`.
- State selection for training: `net.states(brainstate.ParamState)`.

---

## 2. Canonical patterns (copy these shapes; they are verified)

### 2a. Simulation — COBA E/I network (source: `../../examples/brainpy_like/103_COBA_2005.py`)

```python
import brainunit as u
import brainpy, brainstate, braintools

class EINet(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_exc, self.n_inh = 3200, 800
        self.num = self.n_exc + self.n_inh
        self.N = brainpy.state.LIFRef(
            self.num, V_rest=-60.*u.mV, V_th=-50.*u.mV, V_reset=-60.*u.mV,
            tau=20.*u.ms, tau_ref=5.*u.ms,
            V_initializer=braintools.init.Normal(-55., 2., unit=u.mV))
        self.E = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_exc, self.num, conn_num=0.02, conn_weight=0.6*u.mS),
            syn=brainpy.state.Expon.desc(self.num, tau=5.*u.ms),
            out=brainpy.state.COBA.desc(E=0.*u.mV), post=self.N)
        self.I = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_inh, self.num, conn_num=0.02, conn_weight=6.7*u.mS),
            syn=brainpy.state.Expon.desc(self.num, tau=10.*u.ms),
            out=brainpy.state.COBA.desc(E=-80.*u.mV), post=self.N)

    def update(self, t, inp):
        with brainstate.environ.context(t=t):
            spk = self.N.get_spike() != 0.
            self.E(spk[:self.n_exc]); self.I(spk[self.n_exc:]); self.N(inp)
            return self.N.get_spike()

net = EINet()
brainstate.nn.init_all_states(net)
with brainstate.environ.context(dt=0.1*u.ms):
    times = u.math.arange(0.*u.ms, 1000.*u.ms, brainstate.environ.get_dt())
    spikes = brainstate.transform.for_loop(lambda t: net.update(t, 20.*u.mA), times)
# raster: t_idx, n_idx = u.math.where(spikes); plt.scatter(times[t_idx], n_idx, s=1)
```

### 2b. Training — surrogate-gradient SNN (source: `../../examples/brainpy_like/200_surrogate_grad_lif.py`)

```python
import brainpy, brainstate, braintools, brainunit as u

class SNN(brainstate.nn.Module):
    def __init__(self, n_in, n_rec, n_out):
        super().__init__()
        self.i2r = brainstate.nn.Sequential(
            brainstate.nn.Linear(n_in, n_rec,
                w_init=braintools.init.KaimingNormal(unit=u.mA), b_init=braintools.init.ZeroInit(unit=u.mA)),
            brainpy.state.Expon(n_rec, tau=5.*u.ms, g_initializer=braintools.init.Constant(0.*u.mA)))
        self.r = brainpy.state.LIF(n_rec, tau=20*u.ms, V_reset=0*u.mV, V_rest=0*u.mV, V_th=1.*u.mV,
                                   spk_fun=braintools.surrogate.ReluGrad())
        self.r2o = brainstate.nn.Linear(n_rec, n_out, w_init=braintools.init.KaimingNormal())
        self.o = brainpy.state.Expon(n_out, tau=10.*u.ms, g_initializer=braintools.init.Constant(0.))
    def update(self, spike):
        return self.o(self.r2o(self.r(self.i2r(spike))))

with brainstate.environ.context(dt=1.0*u.ms):
    net = SNN(100, 4, 2)
    optimizer = braintools.optim.Adam(lr=3e-3)
    optimizer.register_trainable_weights(net.states(brainstate.ParamState))

    def loss_fn():
        preds = brainstate.transform.for_loop(net.update, x_data)   # [T,B,C]
        preds = u.math.mean(preds, axis=0)
        return braintools.metric.softmax_cross_entropy_with_integer_labels(preds, y_data).mean()

    @brainstate.transform.jit
    def train_step():
        brainstate.nn.init_all_states(net, batch_size=num_sample)
        grads, l = brainstate.transform.grad(loss_fn, net.states(brainstate.ParamState), return_value=True)()
        optimizer.update(grads)
        return l

    for epoch in range(1, 3001):        # outer optimization loop — OK (calls jit'd step)
        loss = train_step()
```

---

## 3. Notebook (`.ipynb`) conventions

- Build uses **MyST-NB** with `nb_execution_mode = "off"` (set by orchestrator).
  So notebooks render from stored outputs; the build will NOT execute them.
  Still write correct, runnable code. If you can execute locally to embed
  outputs, great; if not, leave code cells without outputs (build still passes).
- Author notebooks as JSON `.ipynb` (nbformat 4). First cell = markdown:
  a one-line title (`# Title`), then a short *"What you'll learn / who it's for
  (simulation / training / both)"* paragraph.
- First code cell = imports (`import brainpy, brainstate, braintools` +
  `import brainunit as u` + `import matplotlib.pyplot as plt` as needed).
- Drive all time-stepping with `brainstate.transform` (rule #2).
- End with a markdown **"See also"** cell linking to related pages
  (use ``:doc:`/path` `` style or relative links).
- Minimal `.ipynb` skeleton (each source line is a JSON string in the cell's
  `source` array; code cells need `"outputs": []` and `"execution_count": null`):

```json
{
 "cells": [
  {"cell_type":"markdown","metadata":{},"source":["# Title\n","\n","What you'll learn …"]},
  {"cell_type":"code","execution_count":null,"metadata":{},"outputs":[],
   "source":["import brainpy, brainstate, braintools\n","import brainunit as u\n"]}
 ],
 "metadata": {"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
              "language_info":{"name":"python"}},
 "nbformat": 4, "nbformat_minor": 5
}
```

You can also write notebooks with a small Python helper (build a dict, dump with
`json.dump`) and run it via Bash — whatever is reliable.

---

## 4. Prose (`.rst`) conventions

- Page title underline must be **≥** the title length (RST rule). H1 `===`,
  H2 `---`, H3 `^^^`, H4 `"""`.
- Use `sphinx_design` directives that the theme already loads: `.. grid::` /
  `.. grid-item-card::`, `.. card::`, `.. tab-set::`, `.. dropdown::`,
  `.. admonition::` (but NOT maturity warnings).
- Cross-references: ``:doc:`/concepts/alignpre-alignpost` `` (leading slash =
  from docs root). Math: `$...$` inline / `$$...$$` block (dollarmath enabled),
  or `.. math::`.
- Code blocks: ` .. code-block:: python `.

### Diagram figure snippet (placeholder — user supplies the PNG)

```rst
.. figure:: /_static/concepts/diagram-1-alignpre-alignpost.png
   :alt: AlignPre routes state before the connection matrix (O(N_pre));
         AlignPost routes after it (O(N_post)).
   :width: 90%

   AlignPre vs AlignPost. *(Image generated separately with GPT-image-2; see
   the prompt in the spec §12.)*
```

Place concept diagrams under `docs/_static/concepts/`. Filenames:
`diagram-1-alignpre-alignpost.png`, `diagram-2-memory-complexity.png`,
`diagram-3-bridging-substrate.png`.

---

## 5. Citations (verbatim BibTeX for `project/citing.rst`)

```bibtex
@article{wang2023brainpy,
  title={BrainPy, a flexible, integrative, efficient, and extensible framework for general-purpose brain dynamics programming},
  author={Wang, Chaoming and others},
  journal={eLife}, volume={12}, pages={e86365}, year={2023},
  doi={10.7554/eLife.86365}
}

@inproceedings{wang2024differentiable,
  title={A differentiable brain simulator bridging brain simulation and brain-inspired computing},
  author={Wang, Chaoming and others},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2024}
}
```

Related-work reference for `concepts/online-learning.rst` ONLY (not in citing):
Wang, C., et al. (2026), *Model-agnostic linear-memory online learning in
spiking neural networks*, Nature Communications, doi:10.1038/s41467-026-68453-w.

---

## 6. Where to write things

- **Docs files** → under the worktree:
  `/mnt/d/codes/projects/brainpy-state/.claude/worktrees/docs-rebuild/docs/...`
  (your cwd is the worktree, so relative `docs/...` works).
- **Your plan** → `/mnt/d/codes/projects/brainpy-state/dev/superpowers/plans/<NN>-<name>.md`
  (absolute path; this dir is gitignored and stable — write the plan FIRST).
- **Read the spec** at the absolute path in the header; read your assigned
  section(s) plus this cheatsheet before writing.

---

## 7. Salvage sources (read-only; copy/adapt, do not delete originals)

- `docs/brainpy-guide/{architecture,neurons,synapses,projections}.ipynb`,
  `index.rst`, `network-api.md` — re-cut into concepts/ + brainpy-style/.
- `docs/nest-guide/{index,stdp-divergences}.rst` — re-cut into nest-like/.
- `docs/nest-status/index.rst` — split into nest-like/validation-status +
  integration-categories (strip its Experimental warning per rule #3).
- `docs/quickstart/{installation,5min-tutorial,overview}.{rst,ipynb}` — re-cut
  into get-started/.
- `docs/examples/gallery.rst` — re-tag into examples/brainpy-gallery.rst.
- `examples/brainpy_like/*.py` (~20), `examples/nest_like/*.py` (~75) — the
  scripts the galleries catalogue and tutorials adapt.
