"""Monte Carlo Decision Simulator.

A toolkit for modeling decisions under uncertainty using Monte Carlo simulation,
risk metrics, and sensitivity analysis.
"""

from monte_carlo.distributions import (
    Constant,
    Distribution,
    LogNormal,
    Normal,
    Triangular,
    Uniform,
)

__version__ = "0.1.0"
__all__ = [
    "Distribution",
    "Normal",
    "Uniform",
    "Triangular",
    "LogNormal",
    "Constant",
]
