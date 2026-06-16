# Network API

`brainpy.state.Network` is the foundation for assembling NEST-style
neurons, synapses, and devices into a runnable network. Two entry
points produce the same underlying `brainstate.nn.Module` tree:

- **`brainpy.state.Network`** — subclass it; define populations and
  projections as attributes. Canonical style.
- **`brainpy.state.Builder`** — subclass of `Network` that adds
  `add(name, module)` and `connect(pre, post, *, rule, **kwargs)`
  imperative methods. Convenient for scripts and notebooks.

## Quick start — both styles

```python
import brainstate
import brainunit as u
from brainpy import state as bps

brainstate.environ.set(dt=0.1 * u.ms)

# --- Subclass style -----------------------------------------------------
class TwoPopNet(bps.Network):
    def __init__(self):
        super().__init__()
        self.exc = bps.LIF(800)
        self.inh = bps.LIF(200)
        self.e_to_i = bps.FixedIndegreeProj(
            self.exc, self.inh, K=80,
            weight=0.1 * u.nS,
            syn=bps.Expon.desc(200, tau=5*u.ms),
            out=bps.COBA.desc(E=0*u.mV),
        )

net = TwoPopNet()
brainstate.nn.init_all_states(net)
out = net.simulate(100 * u.ms,
                   monitor={'exc_spike':
                            lambda n: n.exc.get_spike(n.exc.V.value)})

# --- Builder style ------------------------------------------------------
b = bps.Builder()
b.add('exc', bps.LIF(800))
b.add('inh', bps.LIF(200))
b.connect(b.exc, b.inh, rule=bps.FixedIndegreeProj,
          K=80, weight=0.1 * u.nS,
          syn=bps.Expon.desc(200, tau=5*u.ms),
          out=bps.COBA.desc(E=0*u.mV))
brainstate.nn.init_all_states(b)
out = b.simulate(100 * u.ms,
                 monitor={'exc_spike':
                          lambda n: n.exc.get_spike(n.exc.V.value)})
```

Both produce identical module trees and identical simulated output for
the same seed.

## Connection rules

| Class | NEST equivalent |
|---|---|
| `OneToOneProj` | `one_to_one` |
| `AllToAllProj` | `all_to_all` |
| `PairwiseBernoulliProj(p=...)` | `pairwise_bernoulli` |
| `SymmetricPairwiseBernoulliProj(p=...)` | `symmetric_pairwise_bernoulli` |
| `FixedIndegreeProj(K=...)` | `fixed_indegree` |
| `FixedOutdegreeProj(K=...)` | `fixed_outdegree` |
| `FixedTotalNumberProj(N=...)` | `fixed_total_number` |
| `PairwisePoissonProj(mean=...)` | `pairwise_poisson` |

All accept the same uniform keyword set: `weight`, `delay=None` (v1
supports `None` only — `delay=` is deferred), `syn`, `out`,
`allow_autapses=True`, `allow_multapses=True`, `seed=None`.

## Weights

`weight` accepts scalars, arrays of shape `(n_edges,)`, or
`brainpy.state.dist.{Normal, LogNormal, Uniform}` distribution objects
that are sampled **once** at projection `__init__`. This deliberately
differs from NEST's lazy `Parameter` — concrete values are deterministic
given a `seed` and play cleanly with JIT.

```python
proj = bps.FixedIndegreeProj(
    pre, post, K=80,
    weight=bps.dist.Normal(mean=0.1 * u.nS, std=0.01 * u.nS),
    syn=bps.Expon.desc(len(post), tau=5*u.ms),
    out=bps.COBA.desc(E=0*u.mV),
    seed=42,
)
```

## Recording

Two ways to capture state during `simulate()`:

1. **`monitor=` kwarg** — lightweight, returns stacked arrays:

   ```python
   # By attribute path (the attribute must be a State)
   out = net.simulate(100 * u.ms, monitor=['exc.V'])
   V_trace = out['exc.V']   # (T, N_E)

   # By callable — for quantities not stored as a State (e.g. spikes)
   out = net.simulate(100 * u.ms, monitor={
       'spikes': lambda n: n.exc.get_spike(n.exc.V.value),
   })
   ```

2. **`Recorder` + `NESTDevice`** — full NEST-faithful recorder semantics:

   ```python
   from brainpy.state import spike_recorder

   class Net(bps.Network):
       def __init__(self):
           super().__init__()
           self.exc = bps.LIF(800)
           self.rec = bps.Recorder(
               source=self.exc,
               attr=lambda s: s.get_spike(s.V.value),
               device=spike_recorder(in_size=800),
           )
   ```

   `Recorder`'s `attr` accepts a string (read `source.<attr>.value`) or a
   callable. After `simulate()`, access `net.rec.device.events`.

## Stepping by hand

`update()` is canonical — `simulate()` is sugar. Power users can drive
the loop themselves:

```python
import brainstate.transform as transform

times = u.math.arange(0 * u.ms, 100 * u.ms, 0.1 * u.ms)
indices = u.math.arange(times.size)

def step(t, i):
    with brainstate.environ.context(t=t, i=i):
        return net.update(t)

transform.for_loop(step, times, indices)
```
