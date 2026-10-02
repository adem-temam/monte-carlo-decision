"""Tests for sensitivity and tornado analysis."""

import numpy as np
import pytest

from monte_carlo.distributions import Constant, Normal
from monte_carlo.engine import MonteCarloSimulator
from monte_carlo.model import Strategy, UncertainVariable
from monte_carlo.sensitivity import analyze_sensitivity, format_tornado_chart


def test_sensitivity_linear_ranking():
    # Outcome = 10 * x1 + 2 * x2 + 0.1 * x3
    # x1 should have the highest correlation and swing
    x1 = UncertainVariable("major_factor", Normal(50.0, 10.0))
    x2 = UncertainVariable("minor_factor", Normal(50.0, 5.0))
    x3 = UncertainVariable("negligible_factor", Normal(50.0, 1.0))
    x_const = UncertainVariable("constant_factor", Constant(100.0))

    strat = Strategy(
        name="LinearModel",
        variables=[x1, x2, x3, x_const],
        payoff_fn=lambda major_factor, minor_factor, negligible_factor, constant_factor: (
            10.0 * major_factor + 2.0 * minor_factor + 0.1 * negligible_factor + constant_factor
        ),
    )

    sim = MonteCarloSimulator(n_simulations=10_000, seed=42)
    res = sim.run(strat)

    sens = analyze_sensitivity(res)

    assert len(sens) == 4
    # Most influential variable should be ranked first
    assert sens[0].name == "major_factor"
    assert sens[0].correlation > 0.90
    assert sens[0].swing > sens[1].swing

    # Constant variable should be ranked last with 0 correlation
    assert sens[-1].name == "constant_factor"
    assert sens[-1].correlation == 0.0
    assert sens[-1].swing == 0.0


def test_format_tornado_chart():
    x1 = UncertainVariable("demand", Normal(1000.0, 100.0))
    x2 = UncertainVariable("unit_cost", Normal(50.0, 5.0))
    strat = Strategy("Profit", [x1, x2], lambda demand, unit_cost: demand * (100 - unit_cost))

    sim = MonteCarloSimulator(n_simulations=5000, seed=42)
    res = sim.run(strat)

    sens = analyze_sensitivity(res)
    chart = format_tornado_chart(sens)

    assert "SENSITIVITY ANALYSIS" in chart
    assert "demand" in chart
    assert "unit_cost" in chart
    assert "█" in chart
