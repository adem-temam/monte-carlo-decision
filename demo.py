#!/usr/bin/env python3
"""Quickstart Demo: Monte Carlo Decision Simulator.

Compares two engineering processes under uncertainty:
- Process A: Cheaper baseline cost, but slower and higher variance
- Process B: More expensive, but faster and highly consistent

Objective: Score = Cost + 1.5 * Time (Lower total score is better)
"""

from monte_carlo import (
    MonteCarloSimulator,
    Normal,
    Strategy,
    UncertainVariable,
    analyze_sensitivity,
    format_tornado_chart,
)


def main():
    # 1. Define uncertain parameters
    cost_a = UncertainVariable("cost", Normal(mean=100.0, std=10.0), "Manufacturing cost ($)")
    time_a = UncertainVariable("time", Normal(mean=50.0, std=5.0), "Cycle time (minutes)")

    cost_b = UncertainVariable("cost", Normal(mean=120.0, std=4.0), "Manufacturing cost ($)")
    time_b = UncertainVariable("time", Normal(mean=40.0, std=2.0), "Cycle time (minutes)")

    # 2. Define strategies with objective tradeoff: Score = Cost + 1.5 * Time
    lam = 1.5
    strat_a = Strategy("Process A", [cost_a, time_a], lambda cost, time: cost + lam * time)
    strat_b = Strategy("Process B", [cost_b, time_b], lambda cost, time: cost + lam * time)

    # 3. Run Monte Carlo simulation (10,000 iterations)
    sim = MonteCarloSimulator(n_simulations=10_000, seed=42)
    comparison = sim.compare(strat_a, strat_b, higher_is_better=False)

    res_a, res_b = comparison.result_a, comparison.result_b

    # 4. Display findings
    print("=" * 60)
    print("  MONTE CARLO DECISION SIMULATION: Process A vs Process B")
    print("  Objective: Minimize Score = Cost + 1.5 * Time")
    print("=" * 60)
    print(f"Process A -> E[Score]: {res_a.mean:.1f} | Std Dev: {res_a.std:.1f} | 95th Pct: {res_a.percentile(95):.1f}")
    print(f"Process B -> E[Score]: {res_b.mean:.1f} | Std Dev: {res_b.std:.1f} | 95th Pct: {res_b.percentile(95):.1f}")
    print("-" * 60)
    print(f"P(Process A beats Process B): {comparison.win_probability_a * 100:.1f}%")
    print(f"P(Process B beats Process A): {comparison.win_probability_b * 100:.1f}%")
    print("=" * 60)

    # 5. Sensitivity analysis on Process A
    sens_a = analyze_sensitivity(res_a)
    print("\n" + format_tornado_chart(sens_a))


if __name__ == "__main__":
    main()
