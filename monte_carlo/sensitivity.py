"""Sensitivity and Tornado analysis for decision modeling.

Identifies which uncertain input parameters exert the greatest influence on the
decision outcome using rank correlation and quantile swing analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy import stats

from monte_carlo.engine import SimulationResult


@dataclass
class VariableSensitivity:
    """Sensitivity metrics for a single uncertain input variable."""

    name: str
    correlation: float
    p_value: float
    importance_pct: float
    low_outcome: float
    high_outcome: float
    swing: float


def analyze_sensitivity(
    result: SimulationResult,
    percentile_low: float = 10.0,
    percentile_high: float = 90.0,
) -> List[VariableSensitivity]:
    """Analyzes the influence of each input variable on the simulation outcome.

    Calculates:
    1. Spearman Rank Correlation: Quantifies monotonic influence between input and output.
    2. Quantile Swing: Classic decision-analysis metric measuring the change in mean outcome
       when the variable transitions from pessimistic (low percentile) to optimistic (high percentile).

    Args:
        result: SimulationResult containing outcomes and sampled inputs.
        percentile_low: Lower percentile cutoff for swing calculation (default: 10th).
        percentile_high: Upper percentile cutoff for swing calculation (default: 90th).

    Returns:
        List of VariableSensitivity objects sorted by importance (descending).
    """
    outcomes = result.outcomes
    sensitivities: List[VariableSensitivity] = []

    for name, samples in result.inputs.items():
        # Check if the input is constant (zero variance)
        if np.all(samples == samples[0]):
            sensitivities.append(
                VariableSensitivity(
                    name=name,
                    correlation=0.0,
                    p_value=1.0,
                    importance_pct=0.0,
                    low_outcome=float(np.mean(outcomes)),
                    high_outcome=float(np.mean(outcomes)),
                    swing=0.0,
                )
            )
            continue

        # Compute Spearman rank correlation
        res = stats.spearmanr(samples, outcomes)
        corr = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
        pval = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])

        # Compute swing by conditioning outcome on extreme quantiles of the input
        q_low = np.percentile(samples, percentile_low)
        q_high = np.percentile(samples, percentile_high)

        mask_low = samples <= q_low
        mask_high = samples >= q_high

        mean_low = float(np.mean(outcomes[mask_low])) if np.any(mask_low) else float(np.mean(outcomes))
        mean_high = float(np.mean(outcomes[mask_high])) if np.any(mask_high) else float(np.mean(outcomes))
        swing = abs(mean_high - mean_low)

        sensitivities.append(
            VariableSensitivity(
                name=name,
                correlation=corr,
                p_value=pval,
                importance_pct=0.0,  # Computed below after normalizing
                low_outcome=mean_low,
                high_outcome=mean_high,
                swing=swing,
            )
        )

    # Compute relative importance as percentage of total absolute correlation
    total_abs_corr = sum(abs(s.correlation) for s in sensitivities)
    for s in sensitivities:
        if total_abs_corr > 0:
            s.importance_pct = (abs(s.correlation) / total_abs_corr) * 100.0
        else:
            s.importance_pct = 0.0

    # Sort descending by absolute correlation (or swing)
    sensitivities.sort(key=lambda s: abs(s.correlation), reverse=True)
    return sensitivities


def format_tornado_chart(
    sensitivities: List[VariableSensitivity],
    bar_width: int = 20,
) -> str:
    """Formats sensitivity results into a clean ASCII/Unicode Tornado chart."""
    if not sensitivities:
        return "No variable sensitivity data available."

    lines = [
        "-" * 68,
        f"  SENSITIVITY ANALYSIS (TORNADO RANKING)",
        f"  Identifies which assumptions drive outcome variance the most",
        "-" * 68,
        f"{'Variable':<22} | {'Correlation':<12} | {'Swing':<10} | {'Impact':<18}",
        "-" * 68,
    ]

    max_corr = max((abs(s.correlation) for s in sensitivities), default=1.0)
    if max_corr == 0:
        max_corr = 1.0

    for s in sensitivities:
        norm_impact = abs(s.correlation) / max_corr
        num_blocks = int(round(norm_impact * bar_width))
        bar = "█" * num_blocks
        corr_sign = f"{s.correlation:+.3f}"
        lines.append(f"{s.name:<22} | {corr_sign:<12} | {s.swing:<10.2f} | {bar}")

    lines.append("-" * 68)
    return "\n".join(lines)
