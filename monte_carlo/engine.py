"""Vectorized Monte Carlo simulation engine.

Runs simulations across single strategies or head-to-head decision comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from monte_carlo.model import Strategy


@dataclass
class SimulationResult:
    """Encapsulates the output of a Monte Carlo simulation run for a single strategy."""

    strategy_name: str
    outcomes: np.ndarray
    inputs: Dict[str, np.ndarray]
    n_simulations: int

    @property
    def mean(self) -> float:
        """Empirical expected value of the payoff."""
        return float(np.mean(self.outcomes))

    @property
    def std(self) -> float:
        """Sample standard deviation."""
        return float(np.std(self.outcomes, ddof=1))

    @property
    def min(self) -> float:
        """Minimum simulated outcome."""
        return float(np.min(self.outcomes))

    @property
    def max(self) -> float:
        """Maximum simulated outcome."""
        return float(np.max(self.outcomes))

    def percentile(self, q: float) -> float:
        """Compute the q-th percentile (0 <= q <= 100)."""
        return float(np.percentile(self.outcomes, q))


@dataclass
class ComparisonResult:
    """Encapsulates the comparative analysis between two candidate strategies."""

    result_a: SimulationResult
    result_b: SimulationResult
    higher_is_better: bool = True

    @property
    def delta(self) -> np.ndarray:
        """Difference array: Outcomes(A) - Outcomes(B)."""
        return self.result_a.outcomes - self.result_b.outcomes

    @property
    def win_probability_a(self) -> float:
        """Probability that Strategy A produces a strictly better outcome than Strategy B."""
        if self.higher_is_better:
            return float(np.mean(self.result_a.outcomes > self.result_b.outcomes))
        else:
            return float(np.mean(self.result_a.outcomes < self.result_b.outcomes))

    @property
    def win_probability_b(self) -> float:
        """Probability that Strategy B produces a strictly better outcome than Strategy A."""
        if self.higher_is_better:
            return float(np.mean(self.result_b.outcomes > self.result_a.outcomes))
        else:
            return float(np.mean(self.result_b.outcomes < self.result_a.outcomes))

    @property
    def tie_probability(self) -> float:
        """Probability of identical outcomes."""
        return float(np.mean(self.result_a.outcomes == self.result_b.outcomes))

    @property
    def expected_advantage_a(self) -> float:
        """Expected margin of difference in favor of Strategy A."""
        diff = self.delta if self.higher_is_better else -self.delta
        return float(np.mean(diff))


class MonteCarloSimulator:
    """Simulation engine that draws random variables and evaluates strategy payoffs.

    Args:
        n_simulations: Number of Monte Carlo trials (default: 10,000).
        seed: Random seed for deterministic, reproducible results.
    """

    def __init__(self, n_simulations: int = 10_000, seed: Optional[int] = 42) -> None:
        if n_simulations <= 0:
            raise ValueError(f"n_simulations must be positive, got {n_simulations}")
        self.n_simulations = n_simulations
        self.seed = seed

    def _create_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def run(self, strategy: Strategy, rng: Optional[np.random.Generator] = None) -> SimulationResult:
        """Runs the simulation for a single strategy.

        Args:
            strategy: The decision strategy to simulate.
            rng: Optional NumPy random generator.

        Returns:
            SimulationResult containing outcomes and input distributions.
        """
        gen = rng or self._create_rng()
        sampled_inputs: Dict[str, np.ndarray] = {}

        for var_name, var in strategy.variables.items():
            sampled_inputs[var_name] = var.sample(self.n_simulations, rng=gen)

        outcomes = strategy.evaluate(sampled_inputs)

        return SimulationResult(
            strategy_name=strategy.name,
            outcomes=outcomes,
            inputs=sampled_inputs,
            n_simulations=self.n_simulations,
        )

    def compare(
        self,
        strategy_a: Strategy,
        strategy_b: Strategy,
        higher_is_better: bool = True,
        shared_variables: bool = True,
    ) -> ComparisonResult:
        """Simulates and compares two strategies head-to-head.

        When `shared_variables` is True and both strategies reference a variable
        with the exact same name and identical distribution, common random numbers
        are used. This paired sampling technique reduces variance when comparing differences.

        Args:
            strategy_a: First candidate strategy.
            strategy_b: Second candidate strategy.
            higher_is_better: Whether a higher payoff indicates a superior outcome.
            shared_variables: If True, uses common random numbers for shared variables.

        Returns:
            ComparisonResult containing both simulation outputs and relative win rates.
        """
        gen = self._create_rng()

        if shared_variables:
            # Determine any overlapping variables that share identical distributions
            shared_inputs: Dict[str, np.ndarray] = {}
            for name, var_a in strategy_a.variables.items():
                if name in strategy_b.variables:
                    var_b = strategy_b.variables[name]
                    # If same type and matching properties, sample once
                    if type(var_a.distribution) is type(var_b.distribution) and repr(var_a.distribution) == repr(var_b.distribution):
                        shared_inputs[name] = var_a.sample(self.n_simulations, rng=gen)

            # Sample A's remaining inputs
            inputs_a: Dict[str, np.ndarray] = {}
            for name, var in strategy_a.variables.items():
                if name in shared_inputs:
                    inputs_a[name] = shared_inputs[name]
                else:
                    inputs_a[name] = var.sample(self.n_simulations, rng=gen)

            # Sample B's remaining inputs
            inputs_b: Dict[str, np.ndarray] = {}
            for name, var in strategy_b.variables.items():
                if name in shared_inputs:
                    inputs_b[name] = shared_inputs[name]
                else:
                    inputs_b[name] = var.sample(self.n_simulations, rng=gen)

            outcomes_a = strategy_a.evaluate(inputs_a)
            outcomes_b = strategy_b.evaluate(inputs_b)

            res_a = SimulationResult(strategy_a.name, outcomes_a, inputs_a, self.n_simulations)
            res_b = SimulationResult(strategy_b.name, outcomes_b, inputs_b, self.n_simulations)
        else:
            res_a = self.run(strategy_a, rng=gen)
            res_b = self.run(strategy_b, rng=gen)

        return ComparisonResult(
            result_a=res_a,
            result_b=res_b,
            higher_is_better=higher_is_better,
        )
