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


[`brainpy.state`](https://github.com/chaobrain/brainpy.state) provides **spiking neural network models** built on [JAX](https://github.com/jax-ml/jax) and [brainstate](https://github.com/chaobrain/brainstate). It is the point-neuron modeling layer of the [BrainX ecosystem](https://brainmodeling.readthedocs.io/).

The library ships two model families:

- **BrainPy-style models** — high-level, composable neurons (LIF, ALIF, AdEx, HH, Izhikevich, …), synapses (Expon, Alpha, AMPA, GABA, NMDA), projections, readouts, input generators, and short-term plasticity, designed in the tradition of [BrainPy](https://brainpy.readthedocs.io/).
- **NEST-compatible models** — JAX re-implementations of [NEST simulator](https://nest-simulator.readthedocs.io/) neurons, synapses, plasticity (STDP, STP), and devices that preserve NEST's parameter names and unit conventions.

All parameters carry **physical units** via [saiunit](https://github.com/chaobrain/saiunit); BrainPy-style neurons are differentiable through surrogate gradients out of the box.


## Key features

- **Composable architecture** — mix and match neurons, synapses, synaptic outputs (COBA / CUBA / MgBlock), and projections.
- **Physical units everywhere** — parameters use `saiunit` quantities (`mV`, `ms`, `nS`, …), preventing unit errors at construction time.
- **Differentiable** — surrogate gradients enable backpropagation through BrainPy-style spiking networks for gradient-based training.
- **NEST-compatible parameter names** — port models from NEST with minimal friction (NEST-compatible API is experimental, see status below).
- **Hardware-accelerated** — JAX backend with JIT compilation for CPU, GPU, and TPU.


## API status and maturity

`brainpy.state` ships two model families with different maturity levels. **The BrainPy-style API is stable; the NEST-compatible API is experimental.**

| Component                                                              | Status            | Notes |
|------------------------------------------------------------------------|-------------------|-------|
| Base classes (`Dynamics`, `Neuron`, `Synapse`)                         | **Stable**        | Public API stable for the 0.0.x series |
| BrainPy-style neurons (LIF, ALIF, AdEx, HH, Izhikevich, …)             | **Stable**        | 45+ models, fully tested |
| BrainPy-style synapses, projections, readouts, input generators         | **Stable**        | COBA / CUBA / MgBlock, STP / STD, Expon / Alpha / AMPA / GABA / NMDA |
| Surrogate-gradient training                                            | **Stable**        | All BrainPy-style neurons differentiable |
| NEST-compatible neurons (IAF, AdEx, GIF, GLIF, HH, …)                  | **Experimental**  | Under active development; parameter names, defaults, and numerical behavior may change |
| NEST-compatible synapses and plasticity (STDP, STP)                    | **Experimental**  | Under active development |
| NEST-compatible devices (generators, recorders, detectors)             | **Experimental**  | Under active development |
| Rate / binary neurons (NEST set)                                       | **Experimental**  | Subset of NEST set; coverage incomplete |

See the [NEST-style status page](https://brainx.chaobrain.com/brainpy-state/nest-status/index.html) for what is currently available, what is incomplete, and what users should not yet rely on.


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

A small excitatory–inhibitory balanced network using the Stable BrainPy-style API:

```python
import brainpy.state
import brainstate
import braintools
import saiunit as u


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

with brainstate.environ.context(dt=0.1 * u.ms):
    times = u.math.arange(0. * u.ms, 1000. * u.ms, brainstate.environ.get_dt())
    spikes = brainstate.transform.for_loop(
        lambda t: net.update(t, 20. * u.mA), times,
    )
```

See [`examples/103_COBA_2005.py`](https://github.com/chaobrain/brainpy.state/blob/main/examples/103_COBA_2005.py) for the full version with visualization.


## Documentation

- **Docs site**: <https://brainx.chaobrain.com/brainpy-state/>
- **5-minute tutorial**: <https://brainx.chaobrain.com/brainpy-state/quickstart/5min-tutorial.html>
- **BrainPy-style modeling guide**: <https://brainx.chaobrain.com/brainpy-state/brainpy-guide/index.html>
- **API reference**: <https://brainx.chaobrain.com/brainpy-state/api/index.html>
- **NEST-style status**: <https://brainx.chaobrain.com/brainpy-state/nest-status/index.html>


## Examples and tutorials

Runnable examples live in [`examples/`](https://github.com/chaobrain/brainpy.state/tree/main/examples) and are catalogued in the [Examples Gallery](https://brainx.chaobrain.com/brainpy-state/examples/gallery.html). Highlights:

- E-I balanced networks, COBA and CUBA variants
- Gamma oscillations (Susin & Destexhe 2021: AI, CHING, ING, PING mechanisms)
- Surrogate-gradient training on Fashion-MNIST and MNIST
- Joglekar et al. 2018 cortical propagation model


## Development status

`brainpy.state` is in the 0.0.x series. The BrainPy-style API is considered
stable for this series; the NEST-compatible API is under active development.
Pin a specific version (`brainpy.state==0.0.4`) when reproducibility matters.

- **Bug reports**: <https://github.com/chaobrain/brainpy.state/issues>
- **Releases**: <https://github.com/chaobrain/brainpy.state/releases>


## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, coding conventions, testing guidance, and documentation
workflow (including how to document new APIs and how to mark experimental
features).


## Ecosystem

`brainpy.state` is one part of the [BrainX ecosystem](https://brainmodeling.readthedocs.io/):

| Package | Description |
|---------|-------------|
| [brainstate](https://github.com/chaobrain/brainstate) | State management for JAX-based brain modeling |
| [saiunit](https://github.com/chaobrain/saiunit) | Physical units for neuroscience |
| [brainevent](https://github.com/chaobrain/brainevent) | Event-driven sparse operators |
| [braintools](https://github.com/chaobrain/braintools) | Surrogate gradients, analysis, and utilities |


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
