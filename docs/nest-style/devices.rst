.. _nest-devices:

=======
Devices
=======

.. currentmodule:: brainpy.state

Stimulation and recording devices, faithful to NEST's device vocabulary. A
:class:`Simulator` device is created like any model and connected into the
network; recorded data is read back after the run from the
:class:`SimulationResult`.

.. code-block:: python

   import brainunit as u
   from brainpy import state as bp

   sim = bp.Simulator(dt=0.1 * u.ms)
   neuron = sim.create(bp.iaf_psc_exp, 1)
   pg = sim.create(bp.poisson_generator, 1, rate=8000. * u.Hz, rng_seed=0)
   sr = sim.create(bp.spike_recorder)
   mm = sim.create(bp.multimeter, record_from=["V_m"], interval=0.1 * u.ms)

   sim.connect(pg, neuron, weight=10. * u.pA, delay=1. * u.ms)
   sim.connect(neuron, sr)
   sim.connect(mm, neuron)              # reversed: the multimeter observes the neuron

   res = sim.simulate(100. * u.ms)


Two connection directions
=========================

The direction of a device ``connect`` follows NEST:

- **Stimulation devices** are sources: ``connect(generator, neuron)`` — the
  generator drives the neuron.
- **Recording devices** observe: ``connect(voltmeter, neuron)`` /
  ``connect(multimeter, neuron)`` is the *reversed* direction, because the
  recorder samples the neuron rather than receiving events from it. A
  ``spike_recorder`` taps a node's spikes with ``connect(neuron, recorder)``.


Current vs spike sources
========================

A load-bearing distinction:

- **Current generators** inject a current (``pA``) through the neuron's current
  ring buffer — a NEST-faithful one-step delay. These are
  ``dc_generator``, ``ac_generator``, ``noise_generator``,
  ``step_current_generator``, ``step_rate_generator``.
- **Spike sources** deliver delayed delta events: ``poisson_generator`` (and
  ``_ps`` / ``inhomogeneous`` / ``sinusoidal`` variants), ``spike_generator``
  (explicit spike times), ``spike_train_injector``, ``spike_dilutor``, and the
  ``gamma_sup`` / ``mip`` / ``ppd_sup`` / ``pulsepacket`` generators.

A generator **fans out** to one independent train per target neuron, matching
NEST. A multi-channel generator (``create(poisson_generator, k, rate=[...])``)
gives a ``k``-segment view; ``connect`` then takes a per-channel weight vector.


Generators
==========

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Device
     - Use
   * - ``dc_generator``
     - Constant current (``pA``) over a ``[start, stop]`` window.
   * - ``ac_generator``
     - Sinusoidal current.
   * - ``noise_generator``
     - White-noise current, ``mean`` ± ``std`` (``pA``), independent per target.
   * - ``step_current_generator`` / ``step_rate_generator``
     - Piecewise-constant current / rate.
   * - ``poisson_generator`` (+ ``_ps``, ``inhomogeneous``, ``sinusoidal``)
     - Poisson spike trains at a given ``rate`` (``Hz``), gated by
       ``start`` / ``stop`` and keyed by ``rng_seed``.
   * - ``spike_generator``
     - Emit spikes at explicit ``spike_times`` (``ms``).
   * - ``gamma_sup_generator`` / ``sinusoidal_gamma_generator`` / ``mip_generator``
       / ``ppd_sup_generator`` / ``pulsepacket_generator``
     - Specialized / correlated spike sources.


Recorders
=========

.. list-table::
   :header-rows: 1
   :widths: 26 38 36

   * - Device
     - Records
     - Read back with
   * - ``spike_recorder``
     - Per-step spikes of the tapped node.
     - ``res.spikes(sr)`` (T×N counts), ``res.rate(sr)`` (Hz),
       ``res.n_events(sr)``.
   * - ``multimeter``
     - Named analog recordables (``record_from=['V_m', ...]``) at ``interval``.
     - ``res.trace(mm, 'V_m')`` (T×N) and ``res.times``.
   * - ``voltmeter``
     - Membrane potential (a ``multimeter`` specialized to ``V_m``).
     - ``res.trace(vm, 'V_m')``.
   * - ``weight_recorder``
     - Per-edge synaptic weights of a plastic projection.
     - ``res.weight_trace(proj)`` (T×E).


Detectors
=========

``correlation_detector``, ``correlomatrix_detector``,
``correlospinmatrix_detector``, and ``spin_detector`` compute correlation
statistics from recorded trains. Several detectors are imperative host devices
(NumPy-RNG / Python-loop) and run eagerly — obtain the spike data first, then
drive the detector — rather than inside a compiled simulation loop.

``volume_transmitter`` broadcasts a (dopamine) signal to a modulated plasticity
rule; see :doc:`divergences/stdp` for where its parameters live.


.. admonition:: In-memory recording
   :class: seealso

   Recording is **in memory** only — there is no file/ascii backend yet — so
   NEST's ``record_to`` backend axis collapses to the in-memory equivalent. The
   ``time_in_steps`` format axis is reproduced post-hoc by reading recorded
   spikes either as integer step indices or as times in ``ms``. See
   :doc:`tutorials/04-record-and-analyze`.


See also
========

- :doc:`tutorials/01-one-neuron`, :doc:`tutorials/04-record-and-analyze` —
  devices in runnable tutorials.
- :doc:`connectivity` — how devices connect into the network.
- :doc:`/apis/nest-devices` — the full device API with signatures.
- `NEST simulator documentation <https://nest-simulator.readthedocs.io/>`_ — the
  authoritative reference for upstream device semantics.
