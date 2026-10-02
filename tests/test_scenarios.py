"""Tests for realistic decision scenarios."""

import numpy as np
import pytest

from monte_carlo.engine import MonteCarloSimulator
from monte_carlo.scenarios import get_cloud_scenario, get_manufacturing_scenario


def test_manufacturing_scenario():
    strat_a, strat_b, minimize, unit = get_manufacturing_scenario(batch_size=500)
    assert minimize is True
    assert "Cost" in unit

    sim = MonteCarloSimulator(n_simulations=2000, seed=42)
    comp = sim.compare(strat_a, strat_b, higher_is_better=(not minimize))

    assert comp.result_a.mean > 0
    assert comp.result_b.mean > 0
    # Process B has tighter variance due to automation
    assert comp.result_b.std < comp.result_a.std


def test_cloud_scenario():
    strat_onprem, strat_cloud, minimize, unit = get_cloud_scenario()
    assert minimize is True

    sim = MonteCarloSimulator(n_simulations=2000, seed=42)
    comp = sim.compare(strat_onprem, strat_cloud, higher_is_better=(not minimize), shared_variables=True)

    assert comp.result_a.mean > 0
    assert comp.result_b.mean > 0
    assert len(comp.delta) == 2000
