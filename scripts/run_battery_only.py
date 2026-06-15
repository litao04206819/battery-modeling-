#!/usr/bin/env python3
"""Stage 1: drive the DFN battery alone with a synthetic power/current profile.

Validates the battery model in isolation before any motor/rotor coupling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmr.battery.dfn_model import BatteryConfig  # noqa: E402
from bmr.battery.drive import BatterySimulator  # noqa: E402
from bmr.config import load_yaml  # noqa: E402
from bmr.postprocess.plots import plot_system_timeseries, save_system_csv  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "data" / "results"


def main() -> None:
    cfg = load_yaml("battery.yaml")
    config = BatteryConfig.from_dict(cfg)
    sim = BatterySimulator(config, drive_mode="power")

    # Synthetic 10-minute profile: high-power hover then a lower-power cruise.
    t = np.arange(0.0, 600.0 + 1.0, 1.0)
    pack_power = np.where(t < 120.0, 0.8e3, 0.4e3)  # W at the pack terminals

    print(f"Preset: {cfg['preset']} | parameter set: {config.parameter_set}")
    print(f"Pack: {config.pack.series}s{config.pack.parallel}p "
          f"({config.pack.num_cells} cells)")

    result = sim.run_power_profile(t, pack_power)

    print(f"Final SOC: {result.final_soc:.3f}")
    print(f"Min bus voltage: {result.pack_voltage_V.min():.2f} V")
    print(f"Max cell temp: {result.cell_temperature_K.max() - 273.15:.2f} °C")

    save_system_csv(RESULTS / "battery_only.csv", {
        "time_s": result.time_s,
        "pack_power_W": np.interp(result.time_s, t, pack_power),
        "pack_voltage_V": result.pack_voltage_V,
        "pack_current_A": result.pack_current_A,
        "soc": result.soc,
        "cell_temperature_K": result.cell_temperature_K,
    })
    plot_system_timeseries(
        RESULTS / "battery_only.png", result.time_s,
        pack_power_W=np.interp(result.time_s, t, pack_power),
        pack_voltage_V=result.pack_voltage_V,
        pack_current_A=result.pack_current_A,
        soc=result.soc,
        cell_temperature_K=result.cell_temperature_K,
    )
    print(f"Wrote results to {RESULTS}")


if __name__ == "__main__":
    main()
