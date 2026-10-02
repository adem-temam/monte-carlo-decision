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
from monte_carlo.engine import (
    ComparisonResult,
    MonteCarloSimulator,
    SimulationResult,
)
from monte_carlo.metrics import (
    RiskMetrics,
    SummaryStatistics,
    compute_risk_metrics,
    compute_summary,
    format_comparison_summary,
)
from monte_carlo.model import (
    Strategy,
    UncertainVariable,
)
from monte_carlo.plots import (
    plot_decision_dashboard,
    plot_distributions,
    plot_ecdf,
    plot_payoff_delta,
    plot_tornado,
)
from monte_carlo.sensitivity import (
    VariableSensitivity,
    analyze_sensitivity,
    format_tornado_chart,
)

__version__ = "0.1.0"
__all__ = [
    "Distribution",
    "Normal",
    "Uniform",
    "Triangular",
    "LogNormal",
    "Constant",
    "UncertainVariable",
    "Strategy",
    "MonteCarloSimulator",
    "SimulationResult",
    "ComparisonResult",
    "SummaryStatistics",
    "RiskMetrics",
    "compute_summary",
    "compute_risk_metrics",
    "format_comparison_summary",
    "VariableSensitivity",
    "analyze_sensitivity",
    "format_tornado_chart",
    "plot_distributions",
    "plot_ecdf",
    "plot_payoff_delta",
    "plot_tornado",
    "plot_decision_dashboard",
]
