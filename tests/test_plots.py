"""Tests for visualization generation."""

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless test environments
import matplotlib.pyplot as plt
import numpy as np
import pytest

from monte_carlo.distributions import Normal
from monte_carlo.engine import MonteCarloSimulator
from monte_carlo.model import Strategy, UncertainVariable
from monte_carlo.plots import (
    plot_decision_dashboard,
    plot_distributions,
    plot_ecdf,
    plot_payoff_delta,
    plot_tornado,
)
from monte_carlo.sensitivity import analyze_sensitivity


def test_plots_generation():
    x1 = UncertainVariable("x1", Normal(100.0, 10.0))
    x2 = UncertainVariable("x2", Normal(50.0, 5.0))
    strat_a = Strategy("Strategy A", [x1, x2], lambda x1, x2: x1 + x2)

    y1 = UncertainVariable("y1", Normal(110.0, 5.0))
    y2 = UncertainVariable("y2", Normal(45.0, 2.0))
    strat_b = Strategy("Strategy B", [y1, y2], lambda y1, y2: y1 + y2)

    sim = MonteCarloSimulator(n_simulations=1000, seed=42)
    comp = sim.compare(strat_a, strat_b, higher_is_better=True)
    sens_a = analyze_sensitivity(comp.result_a)

    with tempfile.TemporaryDirectory() as tmpdir:
        dashboard_path = Path(tmpdir) / "test_dashboard.png"
        fig = plot_decision_dashboard(comp, sens_a, output_path=dashboard_path)

        assert dashboard_path.exists()
        assert dashboard_path.stat().st_size > 1000
        plt.close(fig)
