"""Tests for probability distribution sampling and statistical properties."""

import numpy as np
import pytest

from monte_carlo.distributions import (
    Constant,
    LogNormal,
    Normal,
    Triangular,
    Uniform,
)


def test_normal_distribution():
    mean, std = 100.0, 15.0
    dist = Normal(mean, std)

    assert dist.mean == mean
    assert dist.std == std
    assert dist.variance == std**2

    rng = np.random.default_rng(42)
    samples = dist.sample(size=100_000, rng=rng)

    assert samples.shape == (100_000,)
    # Verify law of large numbers convergence within 3 sigma of sampling error
    sample_mean = np.mean(samples)
    sample_std = np.std(samples)
    assert pytest.approx(sample_mean, rel=0.01) == mean
    assert pytest.approx(sample_std, rel=0.01) == std

    # Invalid std should raise ValueError
    with pytest.raises(ValueError):
        Normal(mean=10.0, std=-1.0)
    with pytest.raises(ValueError):
        Normal(mean=10.0, std=0.0)


def test_uniform_distribution():
    low, high = 20.0, 80.0
    dist = Uniform(low, high)

    expected_mean = (low + high) / 2.0
    expected_var = ((high - low) ** 2) / 12.0

    assert dist.mean == expected_mean
    assert pytest.approx(dist.variance) == expected_var

    rng = np.random.default_rng(42)
    samples = dist.sample(size=100_000, rng=rng)

    assert samples.min() >= low
    assert samples.max() <= high
    assert pytest.approx(np.mean(samples), rel=0.01) == expected_mean
    assert pytest.approx(np.var(samples), rel=0.01) == expected_var

    # Invalid bounds
    with pytest.raises(ValueError):
        Uniform(low=50.0, high=20.0)
    with pytest.raises(ValueError):
        Uniform(low=50.0, high=50.0)


def test_triangular_distribution():
    low, mode, high = 10.0, 30.0, 60.0
    dist = Triangular(low, mode, high)

    expected_mean = (low + mode + high) / 3.0
    assert pytest.approx(dist.mean) == expected_mean

    rng = np.random.default_rng(42)
    samples = dist.sample(size=100_000, rng=rng)

    assert samples.min() >= low
    assert samples.max() <= high
    assert pytest.approx(np.mean(samples), rel=0.01) == expected_mean
    assert pytest.approx(np.var(samples), rel=0.02) == dist.variance

    # Invalid bounds
    with pytest.raises(ValueError):
        Triangular(low=40.0, mode=30.0, high=50.0)
    with pytest.raises(ValueError):
        Triangular(low=20.0, mode=20.0, high=20.0)


def test_lognormal_distribution():
    mu, sigma = 3.0, 0.25
    dist = LogNormal(mu, sigma)

    rng = np.random.default_rng(42)
    samples = dist.sample(size=100_000, rng=rng)

    assert np.all(samples > 0)
    assert pytest.approx(np.mean(samples), rel=0.02) == dist.mean
    assert pytest.approx(np.var(samples), rel=0.05) == dist.variance

    with pytest.raises(ValueError):
        LogNormal(mu=0.0, sigma=0.0)


def test_constant_distribution():
    val = 42.5
    dist = Constant(val)

    assert dist.mean == val
    assert dist.variance == 0.0
    assert dist.std == 0.0

    samples = dist.sample(100)
    assert np.all(samples == val)


def test_reproducibility_with_seed():
    dist = Normal(50.0, 5.0)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    s1 = dist.sample(1000, rng=rng1)
    s2 = dist.sample(1000, rng=rng2)

    np.testing.assert_array_equal(s1, s2)
