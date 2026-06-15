#!/usr/bin/env python3
"""Stage 3: end-to-end battery-motor-rotor system simulation over a mission.

Two modes:
  --mode profile   (default) build the pack power profile once, then drive the
                   DFN battery with it as a single drive cycle.
  --mode feedback  co-simulate step-by-step with bus-voltage feedback into the
                   motor model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmr.aircraft.airframe import Airframe  # noqa: E402
from bmr.battery.dfn_model import BatteryConfig  # noqa: E402
from bmr.battery.drive import BatterySimulator  # noqa: E402
from bmr.config import load_yaml  # noqa: E402
from bmr.coupling.feedback_loop import run_feedback_simulation  # noqa: E402
from bmr.coupling.profile_builder import build_demand_profile  # noqa: E402
from bmr.mission.profile import Mission  # noqa: E402
from bmr.motor.pmsm import PMSMMotor  # noqa: E402
from bmr.postprocess.plots import plot_system_timeseries, save_system_csv  # noqa: E402
from bmr.rotor.bemt import BEMTRotor  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "data" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["profile", "feedback"], default="profile")
    args = parser.parse_args()

    airframe = Airframe.from_dict(load_yaml("aircraft.yaml"))
    rotor = BEMTRotor(load_yaml("rotor.yaml"), air_density_kg_m3=airframe.air_density_kg_m3)
    motor = PMSMMotor(load_yaml("motor.yaml"))
    mission = Mission.from_dict(load_yaml("mission_hover_cruise.yaml"))
    battery_cfg = BatteryConfig.from_dict(load_yaml("battery.yaml"))

    print(f"Mode: {args.mode} | battery preset: {load_yaml('battery.yaml')['preset']}")

    if args.mode == "profile":
        profile = build_demand_profile(mission, airframe, rotor, motor)
        sim = BatterySimulator(battery_cfg, drive_mode="power")
        result = sim.run_power_profile(profile.time_s, profile.pack_power_W)
        time_s = result.time_s
        pack_power = np.interp(time_s, profile.time_s, profile.pack_power_W)
        pack_voltage = result.pack_voltage_V
        pack_current = result.pack_current_A
        soc = result.soc
        temp = result.cell_temperature_K
    else:
        sim = BatterySimulator(battery_cfg, drive_mode="power")
        coupled = run_feedback_simulation(mission, airframe, rotor, motor, sim)
        time_s = coupled.time_s
        pack_power = coupled.pack_power_W
        pack_voltage = coupled.pack_voltage_V
        pack_current = coupled.pack_current_A
        soc = coupled.soc
        temp = coupled.cell_temperature_K

    print(f"Final SOC:        {soc[-1]:.3f}")
    print(f"Min bus voltage:  {pack_voltage.min():.2f} V")
    print(f"Max cell temp:    {temp.max() - 273.15:.2f} °C")
    print(f"Peak pack power:  {pack_power.max() / 1000:.2f} kW")

    tag = args.mode
    save_system_csv(RESULTS / f"full_system_{tag}.csv", {
        "time_s": time_s,
        "pack_power_W": pack_power,
        "pack_voltage_V": pack_voltage,
        "pack_current_A": pack_current,
        "soc": soc,
        "cell_temperature_K": temp,
    })
    plot_system_timeseries(
        RESULTS / f"full_system_{tag}.png", time_s,
        pack_power_W=pack_power, pack_voltage_V=pack_voltage,
        pack_current_A=pack_current, soc=soc, cell_temperature_K=temp,
    )
    print(f"Wrote results to {RESULTS}")


if __name__ == "__main__":
    main()
