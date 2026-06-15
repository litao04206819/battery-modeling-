"""Integration test for the DFN battery driven by a power profile.

Uses a short profile and the isothermal-friendly high-rate preset to keep the
solve fast; this exercises the real PyBaMM pipeline end-to-end.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmr.battery.dfn_model import BatteryConfig
from bmr.battery.drive import BatterySimulator
from bmr.config import load_yaml

pybamm = pytest.importorskip("pybamm")


def _fast_config():
    cfg = load_yaml("battery.yaml")
    cfg["preset"] = "high_rate_thermal"
    # Small pack keeps the per-cell power sensible for the short test profile.
    cfg["pack"] = {"series": 12, "parallel": 8}
    return BatteryConfig.from_dict(cfg)


def test_constant_power_discharge():
    config = _fast_config()
    sim = BatterySimulator(config, drive_mode="power")

    t = np.arange(0.0, 60.0 + 1.0, 1.0)
    pack_power = np.full_like(t, 1.0e3)  # 1 kW pack discharge

    result = sim.run_power_profile(t, pack_power)

    # SOC decreases under discharge.
    assert result.final_soc < config.initial_soc
    # Pack voltage in a plausible band for a 12s pack of Li-ion cells.
    assert 30.0 < result.pack_voltage_V.min() < 55.0
    # Temperature stays physical.
    assert np.all(result.cell_temperature_K > 273.15)
    assert result.cell_temperature_K.max() < 350.0


def test_pack_scaling():
    """Pack voltage must be the cell voltage times the series count."""
    config = _fast_config()
    sim = BatterySimulator(config, drive_mode="power")
    t = np.arange(0.0, 30.0 + 1.0, 1.0)
    result = sim.run_power_profile(t, np.full_like(t, 800.0))
    ratio = result.pack_voltage_V / result.cell_voltage_V
    assert np.allclose(ratio, config.pack.series, rtol=1e-6)
