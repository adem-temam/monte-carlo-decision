"""Tests for descriptive statistics and risk metrics calculations."""

import numpy as np
import pytest

from monte_carlo.metrics import (
    compute_risk_metrics,
    compute_summary,
    format_comparison_summary,
)


def test_compute_summary():
    data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    stats = compute_summary(data)

    assert stats.mean == 30.0
    assert stats.median == 30.0
    assert stats.min == 10.0
    assert stats.max == 50.0
    assert stats.iqr == 20.0  # 40 - 20
    assert pytest.approx(stats.skewness, abs=1e-5) == 0.0


def test_risk_metrics_maximization():
    # Symmetric standard normal samples
    rng = np.random.default_rng(42)
    data = rng.normal(loc=100.0, scale=10.0, size=50_000)

    # For maximization: VaR at 95% confidence is 5th percentile
    # Theoretical 5th percentile is 100 - 1.645*10 = 83.55
    risk = compute_risk_metrics(data, minimize=False, alpha=0.05, threshold=90.0)

    assert pytest.approx(risk.var_95, rel=0.02) == 83.55
    # CVaR is average of outcomes <= VaR, so it must be strictly lower than VaR
    assert risk.cvar_95 < risk.var_95
    # P(data >= 90) ~ P(Z >= -1.0) ~ 0.8413
    assert risk.prob_threshold is not None
    assert pytest.approx(risk.prob_threshold, abs=0.02) == 0.84


def test_risk_metrics_minimization():
    # Costs: lower is better. Worst 5% are upper tail
    rng = np.random.default_rng(42)
    costs = rng.normal(loc=100.0, scale=10.0, size=50_000)

    # VaR at 95% confidence is 95th percentile
    # Theoretical 95th percentile is 100 + 1.645*10 = 116.45
    risk = compute_risk_metrics(costs, minimize=True, alpha=0.05, threshold=105.0)

    assert pytest.approx(risk.var_95, rel=0.02) == 116.45
    # CVaR is average of outcomes >= VaR, so it must be strictly higher than VaR
    assert risk.cvar_95 > risk.var_95
    # P(costs <= 105) ~ P(Z <= 0.5) ~ 0.6915
    assert risk.prob_threshold is not None
    assert pytest.approx(risk.prob_threshold, abs=0.02) == 0.69


def test_format_comparison_summary():
    data_a = np.array([80, 85, 90, 95, 100], dtype=float)
    data_b = np.array([70, 75, 80, 85, 90], dtype=float)

    stats_a = compute_summary(data_a)
    stats_b = compute_summary(data_b)
    risk_a = compute_risk_metrics(data_a, minimize=False, threshold=85.0)
    risk_b = compute_risk_metrics(data_b, minimize=False, threshold=85.0)

    report = format_comparison_summary(
        "Strategy A", stats_a, risk_a,
        "Strategy B", stats_b, risk_b,
        win_prob_a=0.75,
        expected_advantage_a=10.0,
        minimize=False,
    )

    assert "DECISION ANALYSIS: Strategy A vs Strategy B" in report
    assert "Expected Value E[X]" in report
    assert "Value at Risk" in report
    assert "P(Strategy A outperforms Strategy B): 75.0%" in report
