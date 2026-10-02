"""Decision and risk analysis metrics.

Calculates descriptive statistics, risk measures (VaR, CVaR), and head-to-head
decision comparisons for Monte Carlo outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SummaryStatistics:
    """Core descriptive statistics for a distribution of outcomes."""

    mean: float
    std: float
    median: float
    iqr: float
    min: float
    max: float
    skewness: float


@dataclass
class RiskMetrics:
    """Risk measures quantifying tail risk and confidence bounds."""

    var_95: float
    cvar_95: float
    percentiles: Dict[int, float]
    prob_threshold: Optional[float] = None
    threshold_value: Optional[float] = None
    minimize: bool = False


def compute_summary(outcomes: np.ndarray) -> SummaryStatistics:
    """Computes descriptive statistics for simulated outcomes.

    Args:
        outcomes: 1D array of simulated values.

    Returns:
        SummaryStatistics dataclass.
    """
    arr = np.asarray(outcomes, dtype=np.float64)
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))

    # Skewness calculation (Fisher-Pearson coefficient)
    m2 = float(np.mean((arr - np.mean(arr)) ** 2))
    m3 = float(np.mean((arr - np.mean(arr)) ** 3))
    skew = (m3 / (m2 ** 1.5)) if m2 > 0 else 0.0

    return SummaryStatistics(
        mean=float(np.mean(arr)),
        std=float(np.std(arr, ddof=1)),
        median=float(np.median(arr)),
        iqr=q75 - q25,
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        skewness=skew,
    )


def compute_risk_metrics(
    outcomes: np.ndarray,
    minimize: bool = False,
    alpha: float = 0.05,
    threshold: Optional[float] = None,
) -> RiskMetrics:
    """Computes tail risk measures including Value-at-Risk (VaR) and Conditional VaR (CVaR).

    In decision analysis:
    - If minimizing (e.g. costs, completion time), risk corresponds to the upper tail
      (cost overruns). VaR_95 is the 95th percentile, and CVaR_95 is the average of
      the worst 5% outcomes exceeding that percentile.
    - If maximizing (e.g. profit, revenue), risk corresponds to the lower tail
      (underperformance). VaR_95 is the 5th percentile, and CVaR_95 is the average of
      the worst 5% outcomes below that percentile.

    Args:
        outcomes: 1D array of simulated outcomes.
        minimize: True if lower values are preferable (cost/loss).
        alpha: Tail probability fraction (default: 0.05 for 95% confidence).
        threshold: Optional benchmark value to compute P(success).

    Returns:
        RiskMetrics dataclass.
    """
    arr = np.asarray(outcomes, dtype=np.float64)
    percentile_keys = [5, 10, 25, 50, 75, 90, 95]
    percentile_values = np.percentile(arr, percentile_keys)
    pcts = {k: float(v) for k, v in zip(percentile_keys, percentile_values)}

    if minimize:
        # Upper tail: worst alpha% outcomes are the highest costs
        cutoff_pct = (1.0 - alpha) * 100.0
        var_val = float(np.percentile(arr, cutoff_pct))
        tail = arr[arr >= var_val]
        cvar_val = float(np.mean(tail)) if len(tail) > 0 else var_val
        p_success = float(np.mean(arr <= threshold)) if threshold is not None else None
    else:
        # Lower tail: worst alpha% outcomes are the lowest payoffs
        cutoff_pct = alpha * 100.0
        var_val = float(np.percentile(arr, cutoff_pct))
        tail = arr[arr <= var_val]
        cvar_val = float(np.mean(tail)) if len(tail) > 0 else var_val
        p_success = float(np.mean(arr >= threshold)) if threshold is not None else None

    return RiskMetrics(
        var_95=var_val,
        cvar_95=cvar_val,
        percentiles=pcts,
        prob_threshold=p_success,
        threshold_value=threshold,
        minimize=minimize,
    )


def format_comparison_summary(
    name_a: str,
    stats_a: SummaryStatistics,
    risk_a: RiskMetrics,
    name_b: str,
    stats_b: SummaryStatistics,
    risk_b: RiskMetrics,
    win_prob_a: float,
    expected_advantage_a: float,
    minimize: bool = False,
) -> str:
    """Builds a formatted terminal report comparing two decision strategies."""
    obj_str = "Cost / Loss (Lower is better)" if minimize else "Payoff / Profit (Higher is better)"
    lines = [
        "=" * 68,
        f"  DECISION ANALYSIS: {name_a} vs {name_b}",
        f"  Objective: {obj_str}",
        "=" * 68,
        f"{'Metric':<28} | {name_a:<16} | {name_b:<16}",
        "-" * 68,
        f"{'Expected Value E[X]':<28} | {stats_a.mean:<16.2f} | {stats_b.mean:<16.2f}",
        f"{'Standard Deviation σ':<28} | {stats_a.std:<16.2f} | {stats_b.std:<16.2f}",
        f"{'Median (50th percentile)':<28} | {stats_a.median:<16.2f} | {stats_b.median:<16.2f}",
        f"{'IQR (75th - 25th)':<28} | {stats_a.iqr:<16.2f} | {stats_b.iqr:<16.2f}",
        f"{'Min / Max Range':<28} | {f'{stats_a.min:.1f} .. {stats_a.max:.1f}':<16} | {f'{stats_b.min:.1f} .. {stats_b.max:.1f}':<16}",
        "-" * 68,
        f"{'5th Percentile':<28} | {risk_a.percentiles[5]:<16.2f} | {risk_b.percentiles[5]:<16.2f}",
        f"{'95th Percentile':<28} | {risk_a.percentiles[95]:<16.2f} | {risk_b.percentiles[95]:<16.2f}",
        f"{'Value at Risk (VaR 95%)':<28} | {risk_a.var_95:<16.2f} | {risk_b.var_95:<16.2f}",
        f"{'Conditional VaR (CVaR 95%)':<28} | {risk_a.cvar_95:<16.2f} | {risk_b.cvar_95:<16.2f}",
    ]

    if risk_a.threshold_value is not None:
        lines.append("-" * 68)
        comp_symbol = "<=" if minimize else ">="
        thresh_label = f"P(Outcome {comp_symbol} {risk_a.threshold_value:.1f})"
        p_a = f"{risk_a.prob_threshold * 100:.1f}%" if risk_a.prob_threshold is not None else "N/A"
        p_b = f"{risk_b.prob_threshold * 100:.1f}%" if risk_b.prob_threshold is not None else "N/A"
        lines.append(f"{thresh_label:<28} | {p_a:<16} | {p_b:<16}")

    lines.extend([
        "=" * 68,
        f"  HEAD-TO-HEAD DECISION OUTCOME",
        f"  P({name_a} outperforms {name_b}): {win_prob_a * 100:.1f}%",
        f"  P({name_b} outperforms {name_a}): {(1.0 - win_prob_a) * 100:.1f}%",
        f"  Expected Advantage of {name_a}: {expected_advantage_a:+.2f}",
        "=" * 68,
    ])
    return "\n".join(lines)
