"""Command-line interface for Monte Carlo decision simulation.

Usage:
    python -m monte_carlo.cli --scenario manufacturing --simulations 25000 --save-plots
    python -m monte_carlo.cli --scenario cloud --simulations 20000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from monte_carlo.engine import MonteCarloSimulator
from monte_carlo.metrics import (
    compute_risk_metrics,
    compute_summary,
    format_comparison_summary,
)
from monte_carlo.plots import (
    plot_decision_dashboard,
    plot_distributions,
    plot_ecdf,
    plot_payoff_delta,
    plot_tornado,
)
from monte_carlo.scenarios import (
    get_cloud_scenario,
    get_manufacturing_scenario,
)
from monte_carlo.sensitivity import (
    analyze_sensitivity,
    format_tornado_chart,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monte Carlo Decision Simulator: Compare strategies under uncertainty."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["manufacturing", "cloud"],
        default="manufacturing",
        help="Predefined engineering decision scenario to run (default: manufacturing).",
    )
    parser.add_argument(
        "-n",
        "--simulations",
        type=int,
        default=25_000,
        help="Number of Monte Carlo iterations (default: 25,000).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results (default: 42).",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Generate and save publication-quality visualization figures to disk.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Directory to save generated figures (default: figures).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional budget or target threshold to compute P(success).",
    )
    return parser


def main(args: list[str] | None = None) -> int:
    parser = build_parser()
    opts = parser.parse_args(args)

    if opts.scenario == "manufacturing":
        strat_a, strat_b, minimize, unit_label = get_manufacturing_scenario()
        default_thresh = opts.threshold or 85_000.0
    elif opts.scenario == "cloud":
        strat_a, strat_b, minimize, unit_label = get_cloud_scenario()
        default_thresh = opts.threshold or 7_000.0
    else:
        print(f"Unknown scenario: {opts.scenario}", file=sys.stderr)
        return 1

    print(f"\nRunning Monte Carlo Decision Simulation ({opts.simulations:,} trials, seed={opts.seed})...")
    print(f"Scenario: {opts.scenario.upper()}")
    print(f"Strategy A: {strat_a.name} -> {strat_a.description}")
    print(f"Strategy B: {strat_b.name} -> {strat_b.description}\n")

    simulator = MonteCarloSimulator(n_simulations=opts.simulations, seed=opts.seed)
    higher_is_better = not minimize
    comparison = simulator.compare(
        strat_a, strat_b, higher_is_better=higher_is_better, shared_variables=True
    )

    # Compute descriptive statistics
    stats_a = compute_summary(comparison.result_a.outcomes)
    stats_b = compute_summary(comparison.result_b.outcomes)

    # Compute tail risk metrics
    risk_a = compute_risk_metrics(
        comparison.result_a.outcomes,
        minimize=minimize,
        alpha=0.05,
        threshold=default_thresh,
    )
    risk_b = compute_risk_metrics(
        comparison.result_b.outcomes,
        minimize=minimize,
        alpha=0.05,
        threshold=default_thresh,
    )

    # Print decision comparison table
    summary_report = format_comparison_summary(
        name_a=strat_a.name,
        stats_a=stats_a,
        risk_a=risk_a,
        name_b=strat_b.name,
        stats_b=stats_b,
        risk_b=risk_b,
        win_prob_a=comparison.win_probability_a,
        expected_advantage_a=comparison.expected_advantage_a,
        minimize=minimize,
    )
    print(summary_report)

    # Sensitivity analysis for Strategy A
    sens_a = analyze_sensitivity(comparison.result_a)
    tornado_text = format_tornado_chart(sens_a)
    print(f"\n{tornado_text}")

    if opts.save_plots:
        out_dir = Path(opts.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        dashboard_file = out_dir / f"{opts.scenario}_dashboard.png"
        plot_decision_dashboard(
            comparison=comparison,
            sensitivities_a=sens_a,
            output_path=dashboard_file,
            xlabel=unit_label,
        )
        print(f"\n[✓] Saved 4-panel decision dashboard to: {dashboard_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
