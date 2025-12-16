<p align="center">
  	<img alt="Header image of brainpy.state - brain dynamics programming in Python." src="https://raw.githubusercontent.com/chaobrain/brainpy.state/main/docs/_static/brianpystate_horizontal.png" width=80%>
</p>



<p align="center">
	<a href="https://pypi.org/project/brainpy_state/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/brainpy_state"></a>
	<a href="https://github.com/chaobrain/brainpy.state"><img alt="LICENSE" src="https://img.shields.io/badge/license-Apache%202.0-green?style=plastic"></a>
  	<a href="https://brainpy-state.readthedocs.io/?badge=latest"><img alt="Documentation" src="https://readthedocs.org/projects/brainpy-state/badge/?version=latest"></a>
  	<a href="https://badge.fury.io/py/brainpy_state"><img alt="PyPI version" src="https://badge.fury.io/py/brainpy_state.svg"></a>
    <a href="https://github.com/chaobrain/brainpy.state/actions/workflows/CI.yml"><img alt="Continuous Integration" src="https://github.com/chaobrain/brainpy.state/actions/workflows/CI.yml/badge.svg"></a>
</p>


``brainpy.state`` modernizes [BrainPy](https://github.com/brainpy/BrainPy) simulator of spiking neural networks with state-based syntax in [brainstate](https://github.com/chaobrain/brainstate).
Moreover, ``brainpy.state`` provides more features compared to ``BrainPy``, including:




## Links


- **Source**: https://github.com/chaobrain/brainpy.state
- **Documentation**: https://brainpy-state.readthedocs.io/
- **Bug reports**: https://github.com/chaobrain/brainpy.state/issues
- **Ecosystem**: https://brainmodeling.readthedocs.io/


## Installation

``brainpy.state`` is based on Python (>=3.10) and can be installed on Linux (Ubuntu 16.04 or later), macOS (10.12 or later), and Windows platforms. 

```bash
pip install brainpy-state -U
```

If you want to use ``brainpy.state`` with different hardware support, please install the corresponding version:

```bash
pip install brainpy-state[cpu] -U  # install with CPU support only
pip install brainpy-state[cuda12] -U  # install with CUDA 12.x support
pip install brainpy-state[cuda13] -U  # install with CUDA 13.x support
pip install brainpy-state[tpu] -U  # install with TPU support
```


Install the ``brainpy.state`` with the ecosystem packages:

```bash
pip install BrainX -U
```



## Citing 

If you are using ``brainpy.state``, please consider citing the corresponding paper:

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
```



