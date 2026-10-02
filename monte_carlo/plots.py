"""Publication-quality decision analysis visualizations using Matplotlib.

Produces clean, accessible plots without cluttered styling:
- Distribution overlays (KDE + Expected Value markers)
- Empirical Cumulative Distribution Functions (ECDF) for stochastic dominance
- Head-to-head payoff difference distribution
- Tornado sensitivity charts
- Multi-panel executive decision dashboard
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from monte_carlo.engine import ComparisonResult, SimulationResult
from monte_carlo.sensitivity import VariableSensitivity, analyze_sensitivity

# Muted, accessible color palette
COLOR_A = "#1f77b4"  # Slate Blue
COLOR_B = "#ff7f0e"  # Ochre Orange
COLOR_NEUTRAL = "#2ca02c"  # Sage Green
COLOR_GRID = "#e0e0e0"


def _apply_clean_style(ax: plt.Axes) -> None:
    """Applies clean minimalist styling to an axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
    ax.tick_params(colors="#333333", labelsize=9)


def plot_distributions(
    result_a: SimulationResult,
    result_b: SimulationResult,
    title: str = "Simulated Outcome Distributions",
    xlabel: str = "Payoff / Score",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plots overlaid kernel density estimates and expected value markers for two strategies."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    # Compute KDE for smooth probability density
    res_a, res_b = result_a.outcomes, result_b.outcomes
    min_val = min(np.min(res_a), np.min(res_b))
    max_val = max(np.max(res_a), np.max(res_b))
    x_grid = np.linspace(min_val, max_val, 400)

    kde_a = stats.gaussian_kde(res_a)
    kde_b = stats.gaussian_kde(res_b)

    density_a = kde_a(x_grid)
    density_b = kde_b(x_grid)

    # Plot density curves and shaded fill
    ax.plot(x_grid, density_a, color=COLOR_A, lw=2, label=f"{result_a.strategy_name} (μ={result_a.mean:.1f})")
    ax.fill_between(x_grid, density_a, alpha=0.25, color=COLOR_A)

    ax.plot(x_grid, density_b, color=COLOR_B, lw=2, label=f"{result_b.strategy_name} (μ={result_b.mean:.1f})")
    ax.fill_between(x_grid, density_b, alpha=0.25, color=COLOR_B)

    # Vertical expected value indicators
    ax.axvline(result_a.mean, color=COLOR_A, linestyle="--", lw=1.5, alpha=0.8)
    ax.axvline(result_b.mean, color=COLOR_B, linestyle="--", lw=1.5, alpha=0.8)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Probability Density", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _apply_clean_style(ax)

    return ax


def plot_ecdf(
    result_a: SimulationResult,
    result_b: SimulationResult,
    title: str = "Empirical Cumulative Distribution Function (ECDF)",
    xlabel: str = "Payoff / Score",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plots ECDFs to visualize stochastic dominance and tail risk."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    for res, color in [(result_a, COLOR_A), (result_b, COLOR_B)]:
        sorted_data = np.sort(res.outcomes)
        prob = np.linspace(0.0, 1.0, len(sorted_data))
        ax.plot(sorted_data, prob, color=color, lw=2, label=res.strategy_name)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Cumulative Probability P(X ≤ x)", fontsize=10)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _apply_clean_style(ax)

    return ax


def plot_payoff_delta(
    comparison: ComparisonResult,
    title: str = "Head-to-Head Advantage Distribution (A − B)",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plots the distribution of the outcome margin between strategies."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    delta = comparison.delta
    p_win_a = comparison.win_probability_a * 100
    name_a = comparison.result_a.strategy_name
    name_b = comparison.result_b.strategy_name

    counts, bins, patches = ax.hist(
        delta, bins=50, density=True, color="#4a7c59", alpha=0.6, edgecolor="white", linewidth=0.5
    )

    # Highlight win region
    ax.axvline(0, color="#d9534f", linestyle="-", lw=1.5, alpha=0.9, label="Breakeven (A = B)")
    mean_delta = np.mean(delta)
    ax.axvline(mean_delta, color="#2b5c8f", linestyle="--", lw=1.5, label=f"Mean Advantage ({mean_delta:+.1f})")

    ax.set_title(f"{title}\nP({name_a} wins) = {p_win_a:.1f}%", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(f"Margin ({name_a} − {name_b})", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    _apply_clean_style(ax)

    return ax


def plot_tornado(
    sensitivities: List[VariableSensitivity],
    strategy_name: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plots a horizontal Tornado chart showing variable sensitivity."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    # Reverse order so the most influential appears at the top
    items = list(reversed(sensitivities))
    names = [s.name for s in items]
    corrs = [s.correlation for s in items]
    colors = [COLOR_A if c >= 0 else COLOR_B for c in corrs]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, corrs, color=colors, alpha=0.75, edgecolor="none", height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="#666666", lw=0.8, linestyle="-")

    # Annotate correlation values
    for i, c in enumerate(corrs):
        align = "left" if c >= 0 else "right"
        offset = 0.02 if c >= 0 else -0.02
        ax.text(c + offset, i, f"{c:+.2f}", va="center", ha=align, fontsize=8, color="#333333")

    sub = f" for {strategy_name}" if strategy_name else ""
    ax.set_title(f"Sensitivity Tornado Analysis{sub}\n(Spearman Rank Correlation)", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Correlation with Outcome", fontsize=10)
    ax.set_xlim(-1.15, 1.15)
    _apply_clean_style(ax)

    return ax


def plot_decision_dashboard(
    comparison: ComparisonResult,
    sensitivities_a: List[VariableSensitivity],
    output_path: Optional[str | Path] = None,
    xlabel: str = "Payoff / Score",
) -> plt.Figure:
    """Generates an integrated 4-panel executive decision dashboard figure.

    Panels:
    1. Outcome Probability Density (Overlaid KDE + Mean markers)
    2. Empirical CDF (Stochastic dominance analysis)
    3. Advantage Distribution (Margin of difference A - B)
    4. Tornado Sensitivity Analysis (Primary strategy driver ranking)

    Args:
        comparison: ComparisonResult from the simulation.
        sensitivities_a: Sensitivity results for Strategy A.
        output_path: Optional file path to save the figure (PNG).
        xlabel: Label describing outcome units.

    Returns:
        The generated Matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    name_a = comparison.result_a.strategy_name
    name_b = comparison.result_b.strategy_name

    # Panel 1: Distributions
    plot_distributions(comparison.result_a, comparison.result_b, xlabel=xlabel, ax=axes[0, 0])

    # Panel 2: ECDF
    plot_ecdf(comparison.result_a, comparison.result_b, xlabel=xlabel, ax=axes[0, 1])

    # Panel 3: Delta Distribution
    plot_payoff_delta(comparison, ax=axes[1, 0])

    # Panel 4: Tornado chart
    plot_tornado(sensitivities_a, strategy_name=name_a, ax=axes[1, 1])

    fig.suptitle(
        f"Monte Carlo Decision Dashboard: {name_a} vs {name_b}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight", dpi=180)

    return fig
