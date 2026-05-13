# Missing or Underdeveloped Features

> Part of the editorial report on [`../network-spec-dsl.md`](../network-spec-dsl.md). See [README](./README.md) for navigation.

| Category | What's missing |
|---|---|
| Spatial primitives | Canonical 3D position field; spatial kernels beyond Conv2d; distance-dependent connectivity grounded in stored positions. |
| Morphology | Compartmental / cable-equation models; integration with NEURON/Arbor. (Acceptable as explicit non-goal, but state it.) |
| Plasticity | Third-factor / neuromodulator channels; cross-projection eligibility traces; scheduled plasticity phases; structural plasticity. |
| Stochastic dynamics | Noise terms in neuron / synapse equations; not just stochastic inputs. |
| Experiment protocol | Trial structure, ITI, baselines, warm-up, multi-condition randomization. |
| Datasets | Canonical references, splits, preprocessing. |
| Optimizer / loss / schedule | Currently deferred to user; reproducibility regresses. Either canonicalize or document the trade-off explicitly. |
| DAG composability | Skip connections, parallel branches, merge points at the layer-macro level. |
| Tag-driven and predicate-driven views | `spec.where(tag=...)`, `spec.filter(...)`. |
| Constraint vocabulary | Biophysical priors (parameter coupling, ratios, monotonicity). |
| Hardware constraints | Fan-in/out, core / chip placement hints, quantization vocabulary, time-discretization. |
| Sweep strategies | Random / Sobol / Bayesian sweep beyond cartesian; resume / early stop. |
| Streaming recording | Disk-backed observables; downsampling reducers beyond mean/sum (quantiles, custom callables). |
| Provenance for trained artifacts | A canonical (IR, training-run, trained-parameter-set) bundle with its own hash. |
| Schema evolution | Migration tooling, deprecation policy, version-skew warnings. |
| Profiling / cost models | Memory and compute estimates from the IR (population × density × dt × duration). |
