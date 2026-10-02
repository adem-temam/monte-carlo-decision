"""Tests for strategy definition and simulation engine execution."""

import numpy as np
import pytest

from monte_carlo.distributions import Constant, Normal
from monte_carlo.engine import MonteCarloSimulator
from monte_carlo.model import Strategy, UncertainVariable


def test_strategy_payoff_evaluation():
    v1 = UncertainVariable("cost", Normal(100.0, 10.0))
    v2 = UncertainVariable("time", Normal(50.0, 5.0))

    strat = Strategy(
        name="TestStrategy",
        variables=[v1, v2],
        payoff_fn=lambda cost, time: cost + 2.0 * time,
    )

    inputs = {
        "cost": np.array([100.0, 110.0]),
        "time": np.array([50.0, 45.0]),
    }
    outcomes = strat.evaluate(inputs)

    expected = np.array([100.0 + 2.0 * 50.0, 110.0 + 2.0 * 45.0])
    np.testing.assert_allclose(outcomes, expected)


def test_simulation_reproducibility():
    v = UncertainVariable("x", Normal(50.0, 5.0))
    strat = Strategy("S", [v], lambda x: x * 2.0)

    sim1 = MonteCarloSimulator(n_simulations=5000, seed=123)
    res1 = sim1.run(strat)

    sim2 = MonteCarloSimulator(n_simulations=5000, seed=123)
    res2 = sim2.run(strat)

    np.testing.assert_array_equal(res1.outcomes, res2.outcomes)
    assert res1.mean == pytest.approx(100.0, rel=0.01)
    assert res1.std == pytest.approx(10.0, rel=0.02)


def test_head_to_head_comparison_higher_is_better():
    # Strategy A: Mean 100, Std 5
    # Strategy B: Mean 90, Std 5
    strat_a = Strategy("A", [UncertainVariable("x", Normal(100.0, 5.0))], lambda x: x)
    strat_b = Strategy("B", [UncertainVariable("y", Normal(90.0, 5.0))], lambda y: y)

    sim = MonteCarloSimulator(n_simulations=20_000, seed=42)
    comp = sim.compare(strat_a, strat_b, higher_is_better=True)

    # Difference A - B is Normal(10, sqrt(50) ~= 7.07)
    # P(A > B) = P(Z > -10/7.07 ~= -1.414) ~= 0.921
    assert pytest.approx(comp.win_probability_a, abs=0.02) == 0.92
    assert pytest.approx(comp.win_probability_b, abs=0.02) == 0.08
    assert comp.expected_advantage_a > 9.5


def test_head_to_head_comparison_lower_is_better():
    # Minimizing cost: Process A is cheaper on average
    strat_a = Strategy("Cheaper", [UncertainVariable("c", Normal(80.0, 4.0))], lambda c: c)
    strat_b = Strategy("Costlier", [UncertainVariable("c", Normal(100.0, 4.0))], lambda c: c)

    sim = MonteCarloSimulator(n_simulations=10_000, seed=42)
    comp = sim.compare(strat_a, strat_b, higher_is_better=False)

    # A has lower cost, so A wins in almost all simulations
    assert comp.win_probability_a > 0.99
    assert comp.win_probability_b < 0.01


def test_shared_variables_coupling():
    # Demand is a common variable for both strategies
    demand_var = UncertainVariable("demand", Normal(1000.0, 100.0))
    strat_a = Strategy("A", [demand_var], lambda demand: demand * 10)
    strat_b = Strategy("B", [demand_var], lambda demand: demand * 8)

    sim = MonteCarloSimulator(n_simulations=1000, seed=42)
    comp = sim.compare(strat_a, strat_b, higher_is_better=True, shared_variables=True)

    # Since demand is identical in each paired run: 10*demand > 8*demand for all positive demands
    assert comp.win_probability_a == 1.0
