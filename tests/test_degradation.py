"""Integration test for the cycle-life degradation study.

Uses a single cycle with a short, shallow discharge so the CCCV recharge (and
hence the whole solve) stays fast while still exercising the real OKane2022
coupled-degradation pipeline and summary-variable extraction.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmr.battery.degradation import run_cycle_life
from bmr.battery.dfn_model import BatteryConfig
from bmr.config import load_yaml

pybamm = pytest.importorskip("pybamm")


def _degradation_config():
    cfg = load_yaml("battery.yaml")
    cfg["preset"] = "life_degradation"
    cfg["pack"] = {"series": 12, "parallel": 8}
    return BatteryConfig.from_dict(cfg)


def test_single_cycle_degradation():
    config = _degradation_config()
    # Short, shallow discharge profile.
    t = np.arange(0.0, 60.0 + 1.0, 1.0)
    pack_power = np.full_like(t, 1.0e3)

    result = run_cycle_life(
        config, t, pack_power, n_cycles=1, charge_c_rate=1.0
    )

    assert len(result.soh_percent) >= 1
    # SOH is reported as a percentage and cannot exceed the initial 100 %.
    assert result.soh_percent[0] <= 100.0 + 1e-6
    assert result.final_soh_percent <= 100.0 + 1e-6
    # The coupled-degradation set must expose at least one loss mechanism.
    assert len(result.loss_breakdown_Ah) >= 1
    for vals in result.loss_breakdown_Ah.values():
        assert np.all(vals >= -1e-9)  # capacity losses are non-negative
