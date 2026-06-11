# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Live-NEST parity tests that surface and guard against model bugs.

These tests require a working ``nest`` import and are skipped when it is
unavailable. They drive the ``brainpy.state`` NEST models and the explicit
``Simulator`` API against NEST 3.x reference runs.
"""
