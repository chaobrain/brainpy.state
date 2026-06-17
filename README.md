<p align="center">
  	<img alt="Header image of brainpy.state - brain dynamics programming in Python." src="https://raw.githubusercontent.com/chaobrain/brainpy.state/main/docs/_static/brianpystate_horizontal.png" width=80%>
</p>



<p align="center">
	<a href="https://pypi.org/project/brainpy_state/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/brainpy_state"></a>
	<a href="https://github.com/chaobrain/brainpy.state"><img alt="LICENSE" src="https://img.shields.io/badge/License-Apache%202.0-blue?style=plastic"></a>
  	<a href="https://brainx.chaobrain.com/brainpy-state/"><img alt="Documentation" src="https://img.shields.io/badge/docs-brainx.chaobrain.com-blue"></a>
  	<a href="https://badge.fury.io/py/brainpy_state"><img alt="PyPI version" src="https://badge.fury.io/py/brainpy_state.svg"></a>
    <a href="https://github.com/chaobrain/brainpy.state/actions/workflows/CI.yml"><img alt="Continuous Integration" src="https://github.com/chaobrain/brainpy.state/actions/workflows/CI.yml/badge.svg"></a>
</p>


[`brainpy.state`](https://github.com/chaobrain/brainpy.state) is **one differentiable substrate that bridges brain simulation and brain-inspired computing**. It is the point-neuron modeling layer of the [BrainX ecosystem](https://brainx.chaobrain.com/), built on [JAX](https://github.com/jax-ml/jax) and [brainstate](https://github.com/chaobrain/brainstate).

The same building blocks let you run biophysical spiking networks *and* train them with gradient descent — because the projections that keep simulation memory-efficient are exactly what make gradient-based and online learning memory-efficient too. That bridge was introduced in the [ICLR 2024 paper](https://openreview.net/forum?id=AU2gS9ut61) and is the keystone of the [Core Concepts](https://brainx.chaobrain.com/brainpy-state/concepts/index.html).

`brainpy.state` ships **two model families**, both production-ready:

- **BrainPy-style models** — high-level, composable neurons (LIF, ALIF, AdEx, HH, Izhikevich, …), synapses (Expon, Alpha, AMPA, GABAa, BioNMDA), the AlignPre/AlignPost projections, synaptic outputs (COBA / CUBA / MgBlock), short-term plasticity, readouts, and input generators, in the tradition of [BrainPy](https://brainpy.readthedocs.io/). Differentiable through surrogate gradients out of the box.
- **NEST-compatible models** — JAX re-implementations of [NEST simulator](https://nest-simulator.readthedocs.io/) neurons, synapses, plasticity (STDP, STP), and devices that preserve NEST's parameter names and unit conventions, numerically validated against a live NEST install within documented tolerance bands.

All parameters carry **physical units** via [brainunit](https://github.com/chaobrain/brainunit), catching unit errors at construction time.


## Key features

- **One differentiable substrate** — biophysical simulation and gradient-based / brain-inspired computing share the same neurons, synapses, and projections.
- **Memory-efficient projections** — the AlignPre/AlignPost design aligns synaptic state to the pre- or post-synaptic neurons (`O(N)`) instead of to individual synapses (`O(N_pre · N_post)`), without approximation. See the [AlignPre/AlignPost concept page](https://brainx.chaobrain.com/brainpy-state/concepts/alignpre-alignpost.html).
- **Composable architecture** — mix and match neurons, synapses, synaptic outputs (COBA / CUBA / MgBlock), and projections.
- **Physical units everywhere** — parameters use `brainunit` quantities (`mV`, `ms`, `nS`, …), preventing unit errors at construction time.
- **Differentiable** — surrogate gradients enable backpropagation through spiking networks for gradient-based training.
- **NEST parameter parity** — port models from NEST with minimal friction; the NEST-compatible family is numerically validated against a live NEST install within documented tolerance bands.
- **Hardware-accelerated** — JAX backend with JIT compilation for CPU, GPU, and TPU.


## Installation

`brainpy.state` requires Python ≥ 3.10 and runs on Linux, macOS, and Windows.

```bash
pip install brainpy.state -U
```

For hardware-specific JAX backends:

```bash
pip install brainpy.state[cpu] -U     # CPU only
pip install brainpy.state[cuda12] -U  # CUDA 12.x
pip install brainpy.state[cuda13] -U  # CUDA 13.x
pip install brainpy.state[tpu] -U     # TPU
```

Or install the full BrainX ecosystem:

```bash
pip install BrainX -U
```


## Quickstart

`brainpy.state` offers two entry points, depending on which family you reach for. Both examples build a model from the **public `brainpy.state` namespace** and run it on the JAX backend.

### 1. BrainPy-style — an excitatory–inhibitory balanced network

A COBA E/I network wired with the memory-efficient `AlignPostProj`. The synaptic state lives on the post-synaptic neurons (`O(N_post)`) — see the [AlignPre/AlignPost concept page](https://brainx.chaobrain.com/brainpy-state/concepts/alignpre-alignpost.html) for why this is exact, not an approximation.

```python
import brainpy
import brainstate
import braintools
import brainunit as u


class EINet(brainstate.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_exc, self.n_inh = 3200, 800
        self.num = self.n_exc + self.n_inh
        self.N = brainpy.state.LIFRef(
            self.num,
            V_rest=-60. * u.mV, V_th=-50. * u.mV, V_reset=-60. * u.mV,
            tau=20. * u.ms, tau_ref=5. * u.ms,
            V_initializer=braintools.init.Normal(-55., 2., unit=u.mV),
        )
        self.E = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_exc, self.num,
                                              conn_num=0.02, conn_weight=0.6 * u.mS),
            syn=brainpy.state.Expon.desc(self.num, tau=5. * u.ms),
            out=brainpy.state.COBA.desc(E=0. * u.mV),
            post=self.N,
        )
        self.I = brainpy.state.AlignPostProj(
            comm=brainstate.nn.EventFixedProb(self.n_inh, self.num,
                                              conn_num=0.02, conn_weight=6.7 * u.mS),
            syn=brainpy.state.Expon.desc(self.num, tau=10. * u.ms),
            out=brainpy.state.COBA.desc(E=-80. * u.mV),
            post=self.N,
        )

    def update(self, t, inp):
        with brainstate.environ.context(t=t):
            spk = self.N.get_spike() != 0.
            self.E(spk[:self.n_exc])
            self.I(spk[self.n_exc:])
            self.N(inp)
            return self.N.get_spike()


net = EINet()
brainstate.nn.init_all_states(net)

# Drive the model with brainstate.transform.for_loop — never a bare Python loop.
with brainstate.environ.context(dt=0.1 * u.ms):
    times = u.math.arange(0. * u.ms, 1000. * u.ms, brainstate.environ.get_dt())
    spikes = brainstate.transform.for_loop(
        lambda t: net.update(t, 20. * u.mA), times,
    )
```

See [`examples/brainpy_like/103_COBA_2005.py`](https://github.com/chaobrain/brainpy.state/blob/main/examples/brainpy_like/103_COBA_2005.py) for the full version with a spike raster.

### 2. NEST-like — a single neuron driven by a constant current

The NEST-compatible family uses an explicit `Simulator` with NEST's `create` / `connect` / `simulate` workflow and NEST's parameter names. Here a single `iaf_psc_alpha` is driven by a constant current `I_e` and observed by a `voltmeter` (connected in the reversed direction, as in NEST).

```python
import jax
import brainpy
import brainstate
import brainunit as u

jax.config.update("jax_enable_x64", True)   # NEST-parity runs use float64
brainstate.environ.set(precision=64)

sim = brainpy.state.Simulator(dt=0.1 * u.ms)
neuron = sim.create(brainpy.state.iaf_psc_alpha, 1, I_e=376.0 * u.pA)
vm = sim.create(brainpy.state.voltmeter)
sim.connect(vm, neuron)              # reversed: the voltmeter observes the neuron

res = sim.simulate(1000.0 * u.ms)
v_m = res.trace(vm, "V_m")           # membrane potential trace (with units)
```

See [`examples/nest_like/one_neuron.py`](https://github.com/chaobrain/brainpy.state/blob/main/examples/nest_like/one_neuron.py) for the full version with plotting, and the [NEST-compatible hub](https://brainx.chaobrain.com/brainpy-state/nest-style/index.html) for the tutorials, model directory, and validation evidence.


## Two model families

`brainpy.state` is organized around two model families that share one substrate — explicit `State`, physical units, and `brainstate.transform`-driven simulation. Both are production-ready; the API reference is organized by family for navigation, not by maturity.

### BrainPy-style (native)

The idiomatic, composable layer: 45+ neurons, the AlignPre/AlignPost projections, synaptic outputs, short-term plasticity, readouts, and input generators. Every BrainPy-style neuron is differentiable through surrogate gradients, so the same model you simulate you can also train. Start with the [BrainPy-style modeling guide](https://brainx.chaobrain.com/brainpy-state/brainpy-style/index.html).

### NEST-compatible (validated against live NEST)

JAX re-implementations of NEST neurons, synapses, plasticity rules, and devices that keep NEST's parameter names and unit conventions, so you can port a NEST model with minimal friction. Rather than asking you to take fidelity on trust, every model carries **live-NEST parity evidence**: numerical agreement is checked against a real NEST install within documented tolerance bands. See the [validation showcase](https://brainx.chaobrain.com/brainpy-state/nest-style/validation-status.html) for per-model parity results and tolerance categories.


## Documentation

Full documentation: <https://brainx.chaobrain.com/brainpy-state/>

- **Get started** (install → run → mental model): <https://brainx.chaobrain.com/brainpy-state/get-started/index.html>
- **Core concepts** (the bridging thesis; AlignPre/AlignPost keystone): <https://brainx.chaobrain.com/brainpy-state/concepts/index.html>
- **BrainPy-style modeling**: <https://brainx.chaobrain.com/brainpy-state/brainpy-style/index.html>
- **NEST-compatible hub** (tutorials, models, validation): <https://brainx.chaobrain.com/brainpy-state/nest-style/index.html>
- **API reference**: <https://brainx.chaobrain.com/brainpy-state/apis/index.html>
- **Examples**: <https://brainx.chaobrain.com/brainpy-state/examples/brainpy-gallery.html>


## Examples and tutorials

Runnable examples live in [`examples/`](https://github.com/chaobrain/brainpy.state/tree/main/examples): BrainPy-style scripts under [`examples/brainpy_like/`](https://github.com/chaobrain/brainpy.state/tree/main/examples/brainpy_like) and NEST-compatible ports under [`examples/nest_like/`](https://github.com/chaobrain/brainpy.state/tree/main/examples/nest_like). They are catalogued in the [BrainPy-style gallery](https://brainx.chaobrain.com/brainpy-state/examples/brainpy-gallery.html) and the [NEST gallery](https://brainx.chaobrain.com/brainpy-state/examples/nest-gallery.html). Highlights:

- E-I balanced networks, COBA and CUBA variants
- Gamma oscillations (Susin & Destexhe 2021: AI, CHING, ING, PING mechanisms)
- Surrogate-gradient training on Fashion-MNIST and MNIST
- Joglekar et al. 2018 cortical propagation model
- NEST ports: Brunel random balanced networks, gap-junction networks, astrocyte networks


## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, coding conventions, testing guidance, and documentation
workflow (including how to document new APIs and how to record live-NEST
validation/parity evidence for NEST-compatible models).


## Ecosystem

`brainpy.state` is one part of the [BrainX ecosystem](https://brainx.chaobrain.com/):

| Package | Description |
|---------|-------------|
| [brainstate](https://github.com/chaobrain/brainstate) | State management for JAX-based brain modeling |
| [brainunit](https://github.com/chaobrain/brainunit) | Physical units for neuroscience |
| [brainevent](https://github.com/chaobrain/brainevent) | Event-driven sparse operators |
| [braintools](https://github.com/chaobrain/braintools) | Surrogate gradients, analysis, and utilities |
| [braintrace](https://github.com/chaobrain/braintrace) | Linear-memory online learning for spiking networks |


## Citing

If you use `brainpy.state`, please consider citing the following:

```bibtex
@article {10.7554/eLife.86365,
    article_type = {journal},
    title = {BrainPy, a flexible, integrative, efficient, and extensible framework for general-purpose brain dynamics programming},
    author = {Wang, Chaoming and Zhang, Tianqiu and Chen, Xiaoyu and He, Sichao and Li, Shangyang and Wu, Si},
    editor = {Stimberg, Marcel},
    volume = 12,
    year = 2023,
    month = {dec},
    pub_date = {2023-12-22},
    pages = {e86365},
    citation = {eLife 2023;12:e86365},
    doi = {10.7554/eLife.86365},
    url = {https://doi.org/10.7554/eLife.86365},
    journal = {eLife},
    issn = {2050-084X},
    publisher = {eLife Sciences Publications, Ltd},
}

@inproceedings{wang2024a,
    title={A differentiable brain simulator bridging brain simulation and brain-inspired computing},
    author={Chaoming Wang and Tianqiu Zhang and Sichao He and Hongyaoxing Gu and Shangyang Li and Si Wu},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024},
    url={https://openreview.net/forum?id=AU2gS9ut61}
}
```


## License

`brainpy.state` is released under the Apache License 2.0. See [LICENSE](LICENSE) for the full text.
