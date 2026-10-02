"""Strategy and decision modeling components.

Defines uncertain variables and strategies representing candidate choices under uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np

from monte_carlo.distributions import Distribution


@dataclass
class UncertainVariable:
    """An uncertain parameter characterized by a probability distribution."""

    name: str
    distribution: Distribution
    description: str = ""

    def sample(self, size: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample values for this variable."""
        return self.distribution.sample(size=size, rng=rng)

    def __repr__(self) -> str:
        desc = f" ({self.description})" if self.description else ""
        return f"UncertainVariable({self.name}: {self.distribution}{desc})"


class Strategy:
    """A decision choice with associated uncertain parameters and a payoff / objective function.

    The payoff function defines how the sampled variables combine to produce an outcome
    (e.g., total cost, profit, completion time, or a weighted multi-criteria score).

    Examples:
        >>> cost_var = UncertainVariable("cost", Normal(100, 10))
        >>> time_var = UncertainVariable("time", Normal(50, 5))
        >>> strat = Strategy(
        ...     name="Process A",
        ...     variables=[cost_var, time_var],
        ...     payoff_fn=lambda cost, time: cost + 0.5 * time
        ... )
    """

    def __init__(
        self,
        name: str,
        variables: Sequence[UncertainVariable],
        payoff_fn: Callable[..., Union[np.ndarray, float]],
        description: str = "",
    ) -> None:
        self.name = name
        self.description = description
        self._variables: Dict[str, UncertainVariable] = {v.name: v for v in variables}
        self._payoff_fn = payoff_fn

    @property
    def variables(self) -> Dict[str, UncertainVariable]:
        """Dictionary of variable name -> UncertainVariable."""
        return self._variables

    def evaluate(self, sampled_inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Evaluates the strategy's payoff function using sampled inputs.

        Args:
            sampled_inputs: Dictionary mapping variable names to 1D NumPy arrays of samples.

        Returns:
            1D array of calculated payoff/objective values.
        """
        # Pass only the variables requested by the payoff function
        relevant_inputs = {
            k: v for k, v in sampled_inputs.items() if k in self._variables
        }
        outcomes = self._payoff_fn(**relevant_inputs)

        if np.isscalar(outcomes):
            # If payoff_fn returned a scalar (e.g. constant), broadcast to match input length
            first_arr = next(iter(sampled_inputs.values()))
            return np.full_like(first_arr, fill_value=outcomes, dtype=np.float64)

        return np.asarray(outcomes, dtype=np.float64)

    def __repr__(self) -> str:
        vars_str = ", ".join(self._variables.keys())
        return f"Strategy(name='{self.name}', variables=[{vars_str}])"
