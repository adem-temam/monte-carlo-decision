"""Realistic engineering decision scenarios for Monte Carlo evaluation.

Includes:
1. Manufacturing Process Selection: Low fixed cost/high variance manual process vs high fixed cost/low variance automated CNC.
2. Cloud Infrastructure Architecture: Fixed on-premise cluster vs elastic serverless architecture under uncertain demand.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from monte_carlo.distributions import (
    Constant,
    LogNormal,
    Normal,
    Triangular,
    Uniform,
)
from monte_carlo.model import Strategy, UncertainVariable


def get_manufacturing_scenario(
    batch_size: int = 500,
    delay_penalty_per_hour: float = 40.0,
    target_delivery_hours: float = 1200.0,
) -> Tuple[Strategy, Strategy, bool, str]:
    """Builds the Manufacturing Process Selection scenario.

    Compares:
    - Process A (Conventional Machining): Low setup cost, but high labor variance and scrap rates.
    - Process B (Automated 5-Axis CNC): High upfront capital/tooling cost, but rapid, highly consistent cycle times and minimal scrap.

    Objective:
        Minimize Total Cost = Setup + Material + Labor + Scrap + Delay Penalty.

    Returns:
        (strategy_a, strategy_b, minimize, unit_label)
    """
    # Process A Variables
    cost_setup_a = UncertainVariable("setup_cost", Constant(5_000.0), "Fixed setup and fixturing")
    mat_cost_a = UncertainVariable("unit_material", Uniform(40.0, 55.0), "Material cost per part ($)")
    cycle_time_a = UncertainVariable("cycle_time", Normal(3.2, 0.6), "Machining time per part (hrs)")
    labor_rate_a = UncertainVariable("labor_rate", Constant(35.0), "Machinist hourly wage ($/hr)")
    scrap_rate_a = UncertainVariable("scrap_rate", Triangular(0.03, 0.06, 0.14), "Defect / scrap fraction")

    def payoff_a(setup_cost, unit_material, cycle_time, labor_rate, scrap_rate):
        gross_parts = batch_size / (1.0 - scrap_rate)
        material_total = gross_parts * unit_material
        labor_hours = gross_parts * cycle_time
        labor_total = labor_hours * labor_rate
        # Penalty if total labor hours exceed scheduled delivery deadline
        delay_hours = np.maximum(0.0, labor_hours - target_delivery_hours)
        delay_cost = delay_hours * delay_penalty_per_hour
        return setup_cost + material_total + labor_total + delay_cost

    strat_a = Strategy(
        name="Process A (Conventional)",
        variables=[cost_setup_a, mat_cost_a, cycle_time_a, labor_rate_a, scrap_rate_a],
        payoff_fn=payoff_a,
        description="Conventional manual machining: low setup cost, higher cycle time and scrap risk.",
    )

    # Process B Variables
    cost_setup_b = UncertainVariable("setup_cost", Constant(18_000.0), "High-precision automated CNC tooling")
    mat_cost_b = UncertainVariable("unit_material", Uniform(42.0, 52.0), "Tighter tolerance raw stock ($)")
    cycle_time_b = UncertainVariable("cycle_time", Normal(1.1, 0.12), "CNC automated cycle time (hrs)")
    labor_rate_b = UncertainVariable("labor_rate", Constant(45.0), "CNC technician rate ($/hr)")
    scrap_rate_b = UncertainVariable("scrap_rate", Triangular(0.005, 0.015, 0.04), "Precision CNC defect rate")

    def payoff_b(setup_cost, unit_material, cycle_time, labor_rate, scrap_rate):
        gross_parts = batch_size / (1.0 - scrap_rate)
        material_total = gross_parts * unit_material
        labor_hours = gross_parts * cycle_time
        labor_total = labor_hours * labor_rate
        delay_hours = np.maximum(0.0, labor_hours - target_delivery_hours)
        delay_cost = delay_hours * delay_penalty_per_hour
        return setup_cost + material_total + labor_total + delay_cost

    strat_b = Strategy(
        name="Process B (Automated CNC)",
        variables=[cost_setup_b, mat_cost_b, cycle_time_b, labor_rate_b, scrap_rate_b],
        payoff_fn=payoff_b,
        description="Automated 5-axis CNC machining: high upfront setup, low variance and tight tolerances.",
    )

    return strat_a, strat_b, True, "Total Production Cost ($)"


def get_cloud_scenario() -> Tuple[Strategy, Strategy, bool, str]:
    """Builds the Cloud Infrastructure Architecture scenario.

    Compares:
    - Strategy A (On-Premise Dedicated Cluster): High fixed amortization, low marginal compute, but capacity ceiling risk.
    - Strategy B (Elastic Serverless Cloud): Zero CapEx, purely consumption-based pricing, auto-scaling without bottleneck penalties.

    Objective:
        Minimize Total Monthly Operating Expense ($).

    Returns:
        (strategy_a, strategy_b, minimize, unit_label)
    """
    # Shared demand: monthly incoming requests (in thousands)
    # LogNormal with median ~120,000 requests, right-skewed for traffic spikes
    demand_var = UncertainVariable("demand_k", LogNormal(mu=4.8, sigma=0.45), "Monthly requests (thousands)")

    # Strategy A: On-Premise
    fixed_hw = UncertainVariable("fixed_hardware", Constant(4_500.0), "Amortized server hardware + rack lease")
    power_cooling = UncertainVariable("power_cooling", Uniform(600.0, 1_100.0), "Power and cooling costs")
    admin_hours = UncertainVariable("admin_hours", Normal(25.0, 8.0), "Sysadmin maintenance hours")
    # Capacity limit: 160k requests/month. Overflows incur overflow server cloud burst at $0.08 / k-req
    capacity_limit = 160.0

    def payoff_onprem(demand_k, fixed_hardware, power_cooling, admin_hours):
        admin_cost = np.maximum(0.0, admin_hours) * 75.0  # $75/hr
        overflow_k = np.maximum(0.0, demand_k - capacity_limit)
        overflow_cost = overflow_k * 40.0  # Emergency bursting cost
        return fixed_hardware + power_cooling + admin_cost + overflow_cost

    strat_onprem = Strategy(
        name="On-Prem Dedicated",
        variables=[demand_var, fixed_hw, power_cooling, admin_hours],
        payoff_fn=payoff_onprem,
        description="Dedicated on-premise cluster with fixed lease and capacity ceiling.",
    )

    # Strategy B: Serverless Cloud
    base_cloud_fee = UncertainVariable("base_fee", Constant(150.0), "Cloud platform base fee")
    cost_per_k = UncertainVariable("cost_per_k_req", Triangular(25.0, 32.0, 48.0), "Serverless compute cost per 1k req ($)")
    egress_gb_per_k = UncertainVariable("egress_gb", Normal(1.2, 0.2), "Egress bandwidth GB per 1k req")

    def payoff_cloud(demand_k, base_fee, cost_per_k_req, egress_gb):
        compute_cost = demand_k * cost_per_k_req
        data_transfer = (demand_k * np.maximum(0.1, egress_gb)) * 0.08  # $0.08 / GB egress
        return base_fee + compute_cost + data_transfer

    strat_cloud = Strategy(
        name="Serverless Cloud",
        variables=[demand_var, base_cloud_fee, cost_per_k, egress_gb_per_k],
        payoff_fn=payoff_cloud,
        description="Elastic serverless cloud architecture with zero CapEx and proportional scaling.",
    )

    return strat_onprem, strat_cloud, True, "Monthly Infrastructure Cost ($)"
