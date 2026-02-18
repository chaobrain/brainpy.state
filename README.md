<p align="center">
  	<img alt="Header image of brainpy.state - brain dynamics programming in Python." src="https://raw.githubusercontent.com/chaobrain/brainpy.state/main/docs/_static/brianpystate_horizontal.png" width=80%>
</p>



<p align="center">
	<a href="https://pypi.org/project/brainpy_state/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/brainpy_state"></a>
	<a href="https://github.com/chaobrain/brainpy.state"><img alt="LICENSE" src="https://img.shields.io/badge/License-Apache%202.0-blue?style=plastic"></a>
  	<a href="https://brainpy-state.readthedocs.io/?badge=latest"><img alt="Documentation" src="https://readthedocs.org/projects/brainpy-state/badge/?version=latest"></a>
  	<a href="https://badge.fury.io/py/brainpy_state"><img alt="PyPI version" src="https://badge.fury.io/py/brainpy_state.svg"></a>
    <a href="https://github.com/chaobrain/brainpy.state/actions/workflows/CI.yml"><img alt="Continuous Integration" src="https://github.com/chaobrain/brainpy.state/actions/workflows/CI.yml/badge.svg"></a>
</p>


[`brainpy.state`](https://github.com/chaobrain/brainpy.state) provides 
comprehensive **spiking neural network models** built on [JAX](https://github.com/jax-ml/jax) and [brainstate](https://github.com/chaobrain/brainstate). 
It is the point-neuron modeling layer of the [BrainX ecosystem](https://brainmodeling.readthedocs.io/).

The library ships **167+ models** organized in three tiers:

- **Base classes**: `Dynamics`, `Neuron`, `Synapse`, the abstract foundation every model inherits from.
- **BrainPy-style models (45+)**: high-level, composable neurons (LIF, HH, Izhikevich, …), synapses (Expon, Alpha, AMPA, NMDA, …), projections, readouts, and input generators previously designed in [BrainPy](https://brainpy.readthedocs.io/).
- **NEST-compatible models (119+)**: faithful JAX re-implementations of [NEST simulator](https://nest-simulator.readthedocs.io/) neuron, synapse, plasticity (STDP, STP), and device models.
- All parameters carry **physical units** via [brainunit](https://github.com/chaobrain/brainunit), and every neuron supports surrogate-gradient-based training out of the box.

Compared to `brainpy.dyn`, `brainpy.state` has the following characteristics:

- **Ecosystem compatability**: `brainpy.state` is built on [brainstate](https://github.com/chaobrain/brainstate) and fully compatible with [BrainX ecosystem](https://brainmodeling.readthedocs.io).
- **Model scope**: `brainpy.state` implements much more models including BrainPy-style models plus a large NEST-compatible model set.
- **Scientific ergonomics**: `brainpy.state` uses physical units via `brainunit` by default and is designed for surrogate-gradient training.


## Features

- **Comprehensive model library** — 18 neuron families, 6 synapse types, 9 STDP rules, 17 generators, and more.
- **Physical units everywhere** — parameters use `brainunit` quantities (`mV`, `ms`, `nS`, …), preventing unit errors.
- **Differentiable** — surrogate gradients enable backpropagation through spiking networks for training with gradient descent.
- **NEST compatibility** — benchmarked against [NEST](https://nest-simulator.readthedocs.io/) for numerical accuracy; uses NEST-compatible parameter names.
- **Hardware-accelerated** — JAX backend with JIT compilation for CPU, GPU, and TPU.
- **Composable architecture** — mix-and-match neurons, synapses, synaptic outputs (COBA/CUBA/MgBlock), and projections.


## Quick Example

```python
import brainpy
import brainstate
import brainunit as u

# Create neuron populations
E = brainpy.state.LIF(3200, V_rest=-60*u.mV, V_th=-50*u.mV, tau=20*u.ms)
I = brainpy.state.LIF(800,  V_rest=-60*u.mV, V_th=-50*u.mV, tau=20*u.ms)

```


## Links

- **Documentation**: https://brainpy-state.readthedocs.io/
- **Source**: https://github.com/chaobrain/brainpy.state
- **Bug reports**: https://github.com/chaobrain/brainpy.state/issues
- **Ecosystem**: https://brainmodeling.readthedocs.io/


## Installation

`brainpy.state` requires Python >= 3.10 and runs on Linux, macOS, and Windows.

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


## Ecosystem

`brainpy.state` is one part of the [BrainX ecosystem](https://brainmodeling.readthedocs.io/):

| Package | Description |
|---------|-------------|
| [brainstate](https://github.com/chaobrain/brainstate) | State management for JAX-based brain modeling |
| [brainunit](https://github.com/chaobrain/brainunit) | Physical units for neuroscience |
| [brainevent](https://github.com/chaobrain/brainevent) | Event-driven sparse operators |
| [braintools](https://github.com/chaobrain/braintools) | Surrogate gradients, analysis, and utilities |


## Citing

If you use `brainpy.state`, please consider citing the following paper:

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
    abstract = {Elucidating the intricate neural mechanisms underlying brain functions requires integrative brain dynamics modeling. To facilitate this process, it is crucial to develop a general-purpose programming framework that allows users to freely define neural models across multiple scales, efficiently simulate, train, and analyze model dynamics, and conveniently incorporate new modeling approaches. In response to this need, we present BrainPy. BrainPy leverages the advanced just-in-time (JIT) compilation capabilities of JAX and XLA to provide a powerful infrastructure tailored for brain dynamics programming. It offers an integrated platform for building, simulating, training, and analyzing brain dynamics models. Models defined in BrainPy can be JIT compiled into binary instructions for various devices, including Central Processing Unit (CPU), Graphics Processing Unit (GPU), and Tensor Processing Unit (TPU), which ensures high running performance comparable to native C or CUDA. Additionally, BrainPy features an extensible architecture that allows for easy expansion of new infrastructure, utilities, and machine-learning approaches. This flexibility enables researchers to incorporate cutting-edge techniques and adapt the framework to their specific needs},
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
