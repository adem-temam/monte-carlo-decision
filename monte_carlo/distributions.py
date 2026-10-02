"""Probability distributions for Monte Carlo simulation.

Provides standard distributions for modeling uncertain parameters with vectorized sampling
via modern NumPy random generators.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Distribution(ABC):
    """Abstract base class for probability distributions."""

    @abstractmethod
    def sample(self, size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw random samples from the distribution.

        Args:
            size: Number of samples to generate.
            rng: Optional NumPy Generator for reproducible randomness.

        Returns:
            1D NumPy array of generated float samples.
        """
        pass

    @property
    @abstractmethod
    def mean(self) -> float:
        """Theoretical expected value."""
        pass

    @property
    @abstractmethod
    def variance(self) -> float:
        """Theoretical variance."""
        pass

    @property
    def std(self) -> float:
        """Theoretical standard deviation."""
        return math.sqrt(self.variance)


class Normal(Distribution):
    """Gaussian / Normal distribution.

    Used when uncertain values cluster symmetrically around a central tendency,
    such as process variation, measurement error, or market returns.
    """

    def __init__(self, mean: float, std: float) -> None:
        if std <= 0:
            raise ValueError(f"Standard deviation must be strictly positive, got {std}")
        self._mean = float(mean)
        self._std = float(std)

    def sample(self, size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        gen = rng or np.random.default_rng()
        return gen.normal(loc=self._mean, scale=self._std, size=size)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        return self._std ** 2

    @property
    def std(self) -> float:
        return self._std

    def __repr__(self) -> str:
        return f"Normal(mean={self._mean}, std={self._std})"


class Uniform(Distribution):
    """Continuous Uniform distribution.

    Used when an uncertain parameter is equally likely to fall anywhere within
    a bounded range [low, high].
    """

    def __init__(self, low: float, high: float) -> None:
        if low >= high:
            raise ValueError(f"Low bound ({low}) must be strictly less than high bound ({high})")
        self._low = float(low)
        self._high = float(high)

    def sample(self, size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        gen = rng or np.random.default_rng()
        return gen.uniform(low=self._low, high=self._high, size=size)

    @property
    def mean(self) -> float:
        return (self._low + self._high) / 2.0

    @property
    def variance(self) -> float:
        return ((self._high - self._low) ** 2) / 12.0

    def __repr__(self) -> str:
        return f"Uniform(low={self._low}, high={self._high})"


class Triangular(Distribution):
    """Triangular distribution.

    Widely used in engineering project estimation and three-point PERT analysis
    where domain experts estimate optimistic (low), most likely (mode), and pessimistic (high) values.
    """

    def __init__(self, low: float, mode: float, high: float) -> None:
        if not (low <= mode <= high):
            raise ValueError(f"Requires low <= mode <= high, got low={low}, mode={mode}, high={high}")
        if low == high:
            raise ValueError("Low and high bounds cannot be equal in Triangular distribution")
        self._low = float(low)
        self._mode = float(mode)
        self._high = float(high)

    def sample(self, size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        gen = rng or np.random.default_rng()
        return gen.triangular(left=self._low, mode=self._mode, right=self._high, size=size)

    @property
    def mean(self) -> float:
        return (self._low + self._mode + self._high) / 3.0

    @property
    def variance(self) -> float:
        a, c, b = self._low, self._mode, self._high
        return (a**2 + b**2 + c**2 - a*b - a*c - b*c) / 18.0

    def __repr__(self) -> str:
        return f"Triangular(low={self._low}, mode={self._mode}, high={self._high})"


class LogNormal(Distribution):
    """Log-Normal distribution.

    Used for strictly positive, right-skewed quantities such as task completion times,
    repair durations, or unit costs.

    Parameters refer to the mean (mu) and standard deviation (sigma) of the underlying
    normal variable ln(X).
    """

    def __init__(self, mu: float, sigma: float) -> None:
        if sigma <= 0:
            raise ValueError(f"Sigma must be strictly positive, got {sigma}")
        self._mu = float(mu)
        self._sigma = float(sigma)

    def sample(self, size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        gen = rng or np.random.default_rng()
        return gen.lognormal(mean=self._mu, sigma=self._sigma, size=size)

    @property
    def mean(self) -> float:
        return math.exp(self._mu + (self._sigma ** 2) / 2.0)

    @property
    def variance(self) -> float:
        return (math.exp(self._sigma ** 2) - 1.0) * math.exp(2.0 * self._mu + self._sigma ** 2)

    def __repr__(self) -> str:
        return f"LogNormal(mu={self._mu}, sigma={self._sigma})"


class Constant(Distribution):
    """Deterministic constant pseudo-distribution.

    Useful for baseline fixed parameters that do not have random variation.
    """

    def __init__(self, value: float) -> None:
        self._value = float(value)

    def sample(self, size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        return np.full(size, self._value, dtype=np.float64)

    @property
    def mean(self) -> float:
        return self._value

    @property
    def variance(self) -> float:
        return 0.0

    def __repr__(self) -> str:
        return f"Constant({self._value})"
