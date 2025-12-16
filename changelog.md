# Changelog

## Version 0.0.1

*Initial release of brainpy.state*

This is the first release of `brainpy.state`, which modernizes [BrainPy](https://github.com/brainpy/BrainPy) simulator of spiking neural networks with state-based syntax in [brainstate](https://github.com/chaobrain/brainstate).

### Features

#### Neuron Models

- **Integrate-and-Fire Neurons**
  - `IF`: Basic integrate-and-fire neuron
  - `LIF`, `LIFRef`: Leaky integrate-and-fire neuron (with optional refractory period)
  - `ExpIF`, `ExpIFRef`: Exponential integrate-and-fire neuron
  - `AdExIF`, `AdExIFRef`: Adaptive exponential integrate-and-fire neuron
  - `ALIF`: Adaptive leaky integrate-and-fire neuron
  - `QuaIF`: Quadratic integrate-and-fire neuron
  - `AdQuaIF`, `AdQuaIFRef`: Adaptive quadratic integrate-and-fire neuron
  - `Gif`, `GifRef`: Generalized integrate-and-fire neuron

- **Hodgkin-Huxley Type Neurons**
  - `HH`: Classic Hodgkin-Huxley neuron model
  - `MorrisLecar`: Morris-Lecar neuron model
  - `WangBuzsakiHH`: Wang-Buzsaki modified Hodgkin-Huxley model

- **Izhikevich Neurons**
  - `Izhikevich`, `IzhikevichRef`: Izhikevich neuron model (with optional refractory period)

#### Synapse Models

- **Exponential Synapses**
  - `Expon`: Single exponential decay synapse
  - `DualExpon`: Dual exponential (rise and decay) synapse

- **Receptor-based Synapses**
  - `Alpha`: Alpha function synapse
  - `AMPA`: AMPA receptor synapse
  - `GABAa`: GABAa receptor synapse
  - `BioNMDA`: Biological NMDA receptor synapse

- **Short-term Plasticity**
  - `STP`: Short-term plasticity (facilitation and depression)
  - `STD`: Short-term depression

#### Infrastructure

- Base classes: `Neuron`, `Synapse` for building custom models
- Projection utilities for network connectivity
- Synaptic output handlers (`COBA`, `CUBA`, `MgBlock`)
- Input generation utilities (`SpikeTime`, `PoissonSpike`)
- Readout layers (`Readout`, `LeakyReadout`, `WeightedReadout`)
- Compatibility check for brainpy version (requires >= 2.7.4 or no brainpy installed)

### Dependencies

- Python >= 3.10
- brainstate >= 0.2.0
- brainunit
- brainevent >= 0.0.4
- braintools >= 0.0.9
- jax
- numpy >= 1.15
