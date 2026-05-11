.. warning::

   **Experimental — In Development.** The NEST-compatible model family is
   under active development. Parameter names, defaults, numerical behavior,
   and the set of available models may change without notice across 0.0.x
   releases. See the :doc:`NEST-style status page </nest-status/index>` for
   current scope and limitations.


NEST-Compatible Devices
=======================

Stimulation and recording devices compatible with the `NEST simulator <https://nest-simulator.readthedocs.io/>`_.

.. currentmodule:: brainpy.state


Current Generators
------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    dc_generator
    ac_generator
    noise_generator
    step_current_generator
    step_rate_generator


Spike Generators
----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    spike_generator
    spike_train_injector
    spike_dilutor


Poisson Generators
------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    poisson_generator
    poisson_generator_ps
    inhomogeneous_poisson_generator
    sinusoidal_poisson_generator


Other Generators
----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    gamma_sup_generator
    sinusoidal_gamma_generator
    mip_generator
    ppd_sup_generator
    pulsepacket_generator


Recording Devices
-----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    multimeter
    spike_recorder
    weight_recorder


Detectors
---------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    correlation_detector
    correlomatrix_detector
    correlospinmatrix_detector
    spin_detector


Other Devices
-------------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: classtemplate.rst

    volume_transmitter
