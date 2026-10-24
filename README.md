<p align="center">
  	<img alt="brainspike logo" src="https://raw.githubusercontent.com/chaobrain/brainspike/main/docs/_static/brainspike.png" width=40%>
</p>



<p align="center">
	<a href="https://pypi.org/project/brainspike/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/brainspike"></a>
	<a href="https://github.com/chaobrain/brainspike"><img alt="LICENSE" src="https://img.shields.io/badge/License-Apache%202.0-blue?style=plastic"></a>
  	<a href="https://brainx.chaobrain.com/brainspike/"><img alt="Documentation" src="https://img.shields.io/badge/docs-brainx.chaobrain.com-blue"></a>
  	<a href="https://badge.fury.io/py/brainspike"><img alt="PyPI version" src="https://badge.fury.io/py/brainspike.svg"></a>
    <a href="https://github.com/chaobrain/brainspike/actions/workflows/CI.yml"><img alt="Continuous Integration" src="https://github.com/chaobrain/brainspike/actions/workflows/CI.yml/badge.svg"></a>
</p>


[`brainspike`](https://github.com/chaobrain/brainspike) is the point-neuron spiking-network layer of the [BrainX ecosystem](https://brainx.chaobrain.com/): building, simulating, and training spiking neural networks of point neurons. A new kernel language unifies how models are built, how they are simulated, and how they are trained. It is built on [JAX](https://github.com/jax-ml/jax) and [brainstate](https://github.com/chaobrain/brainstate).

`brainspike` provides:

- **NEST-compatible models** — JAX re-implementations of [NEST simulator](https://nest-simulator.readthedocs.io/) neurons, synapses, plasticity (STDP, STP), and devices that preserve NEST's parameter names and unit conventions, numerically validated against a live NEST install within documented tolerance bands, driven through a NEST-style `Simulator` (`create` / `connect` / `simulate`).
- **A unified kernel** (in development, still private) — one `Network` graph (populations, projections, devices, observers) over models with written contracts (declared ports, states, parameters, and spike rules), so the graph a simulation runs is the graph a learner trains ([architecture spec](docs/specs/2026-10-21-architecture-review-and-redesign.md), [roadmap](docs/specs/2026-10-21-roadmap.md)).

All parameters carry **physical units** via [brainunit](https://github.com/chaobrain/brainunit), catching unit errors at construction time.


## Key features

- **NEST parameter parity** — port models from NEST with minimal friction; every model carries live-NEST parity evidence within documented tolerance bands.
- **Physical units everywhere** — parameters use `brainunit` quantities (`mV`, `ms`, `nS`, …), preventing unit errors at construction time.
- **One graph for simulation and learning** — the new kernel compiles the same `Network` for simulation and for gradient-based or online learning.
- **Hardware-accelerated** — JAX backend with JIT compilation for CPU, GPU, and TPU.


## Scope

`brainspike` is for computational neuroscientists who want NEST models on GPU/TPU or want to train them, and for brain-inspired-computing researchers who need biologically constrained spiking networks.

- **Model scope:** point neurons only (no multi-compartment or population/mean-field models) and no single-unit rate neurons.
- **Out of scope:** detailed multi-compartment morphology, MPI cluster-scale simulation (single-host multi-device is in scope), MUSIC/SONATA co-simulation (SONATA import is in scope), and e-prop (see [braintrace](https://github.com/chaobrain/braintrace)).

**Kernel redesign in progress.** The first kernel milestones have passed their go/no-go gates; the new kernel is still private. Meanwhile the current NEST-style `Simulator` API shown below is frozen (it keeps working; no new features or fixes) and will become a thin layer over the kernel's `Network`. CPU performance of the new kernel has been measured on one benchmark; GPU performance has not been measured yet.

See [Positioning and scope](https://brainx.chaobrain.com/brainspike/project/positioning.html).


## Installation

`brainspike` requires Python ≥ 3.11 and runs on Linux, macOS, and Windows.

```bash
pip install brainspike -U
```

For hardware-specific JAX backends:

```bash
pip install brainspike[cpu] -U     # CPU only
pip install brainspike[cuda12] -U  # CUDA 12.x
pip install brainspike[cuda13] -U  # CUDA 13.x
pip install brainspike[tpu] -U     # TPU
```

Or install the full BrainX ecosystem:

```bash
pip install BrainX -U
```


## Quickstart

A single `iaf_psc_alpha` driven by a constant current `I_e` and observed by a `voltmeter` (connected in the reversed direction, as in NEST):

```python
import jax
import brainstate
import brainunit as u
import brainspike

jax.config.update("jax_enable_x64", True)   # NEST-parity runs use float64
brainstate.environ.set(precision=64)

sim = brainspike.Simulator(dt=0.1 * u.ms)
neuron = sim.create(brainspike.iaf_psc_alpha, 1, I_e=376.0 * u.pA)
vm = sim.create(brainspike.voltmeter)
sim.connect(vm, neuron)              # reversed: the voltmeter observes the neuron

res = sim.simulate(1000.0 * u.ms)
v_m = res.trace(vm, "V_m")           # membrane potential trace (with units)
```

See [`examples/nest_like/one_neuron.py`](https://github.com/chaobrain/brainspike/blob/main/examples/nest_like/one_neuron.py) for the full version with plotting, and the [NEST-compatible hub](https://brainx.chaobrain.com/brainspike/nest-style/index.html) for the tutorials, model directory, and validation evidence.


## Documentation

Full documentation: <https://brainx.chaobrain.com/brainspike/>

- **Get started**: <https://brainx.chaobrain.com/brainspike/get-started/index.html>
- **NEST-compatible hub** (tutorials, models, validation): <https://brainx.chaobrain.com/brainspike/nest-style/index.html>
- **API reference**: <https://brainx.chaobrain.com/brainspike/apis/index.html>
- **Examples**: <https://brainx.chaobrain.com/brainspike/examples/nest-gallery.html>


## Examples

Runnable NEST-compatible ports live in [`examples/nest_like/`](https://github.com/chaobrain/brainspike/tree/main/examples/nest_like) and are catalogued in the [NEST gallery](https://brainx.chaobrain.com/brainspike/examples/nest-gallery.html). Highlights: Brunel random balanced networks, gap-junction networks, astrocyte networks, and STDP / STP plasticity demos.


## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, coding conventions, testing guidance, and documentation
workflow (including how to document new APIs and how to record live-NEST
validation/parity evidence for NEST-compatible models).


## Ecosystem

`brainspike` is one part of the [BrainX ecosystem](https://brainx.chaobrain.com/):

| Package | Description |
|---------|-------------|
| [brainstate](https://github.com/chaobrain/brainstate) | State management for JAX-based brain modeling |
| [brainunit](https://github.com/chaobrain/brainunit) | Physical units for neuroscience |
| [brainevent](https://github.com/chaobrain/brainevent) | Event-driven sparse operators |
| [braintools](https://github.com/chaobrain/braintools) | Surrogate gradients, analysis, and utilities |
| [braintrace](https://github.com/chaobrain/braintrace) | Linear-memory online learning for spiking networks |


## Citing

If you use `brainspike`, please consider citing the following:

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

`brainspike` is released under the Apache License 2.0. See [LICENSE](LICENSE) for the full text.
