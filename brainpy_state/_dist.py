# Copyright 2026 BrainX Ecosystem Limited. Apache 2.0.
"""Distribution objects sampled once at projection init.

This is the brainpy.state-native parameter-randomization API. It
deliberately differs from NEST's lazy ``Parameter`` (which evaluates
at ``Connect`` time): our distributions are sampled eagerly during
projection construction, so the JIT trace sees concrete arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import saiunit as u

__all__ = ['Distribution', 'Normal', 'LogNormal', 'Uniform']


class Distribution:
    """Abstract base. Subclasses implement ``sample(shape, key)``."""
    __module__ = 'brainpy.state.dist'

    def sample(self, shape, key):  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class Normal(Distribution):
    mean: float
    std: float

    def sample(self, shape, key):
        mean_val, mean_unit = u.split_mantissa_unit(self.mean)
        std_val, std_unit = u.split_mantissa_unit(self.std)
        if mean_unit != std_unit:
            raise ValueError(
                f'mean and std must share units, got {mean_unit} and {std_unit}'
            )
        samples = mean_val + std_val * jax.random.normal(key, shape)
        if mean_unit == u.UNITLESS:
            return samples
        return u.maybe_decimal(u.Quantity(samples, unit=mean_unit))


@dataclass
class LogNormal(Distribution):
    mean: float
    std: float

    def sample(self, shape, key):
        return jnp.exp(self.mean + self.std * jax.random.normal(key, shape))


@dataclass
class Uniform(Distribution):
    low: float
    high: float

    def sample(self, shape, key):
        low_val, low_unit = u.split_mantissa_unit(self.low)
        high_val, high_unit = u.split_mantissa_unit(self.high)
        if low_unit != high_unit:
            raise ValueError(
                f'low and high must share units, got {low_unit} and {high_unit}'
            )
        u01 = jax.random.uniform(key, shape)
        samples = low_val + (high_val - low_val) * u01
        if low_unit == u.UNITLESS:
            return samples
        return u.maybe_decimal(u.Quantity(samples, unit=low_unit))


# Override __module__ after class definitions to avoid dataclass type-
# resolution looking up a non-existent ``brainpy.state.dist`` module.
Normal.__module__ = 'brainpy.state.dist'
LogNormal.__module__ = 'brainpy.state.dist'
Uniform.__module__ = 'brainpy.state.dist'
