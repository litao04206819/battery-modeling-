#!/usr/bin/env python3
"""Run a multi-cycle capacity-fade study on the coupled mission.

Builds the rotor+motor pack power demand for one mission, then repeats it as a
discharge/recharge cycle ``--cycles`` times with the life_degradation preset and
reports the state-of-health trend and per-mechanism capacity loss.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from bmr.aircraft.airframe import Airframe  # noqa: E402
from bmr.battery.degradation import run_cycle_life  # noqa: E402
from bmr.battery.dfn_model import BatteryConfig  # noqa: E402
from bmr.config import load_yaml  # noqa: E402
from bmr.coupling.profile_builder import build_demand_profile  # noqa: E402
from bmr.mission.profile import Mission  # noqa: E402
from bmr.motor.pmsm import PMSMMotor  # noqa: E402
from bmr.postprocess.plots import save_system_csv  # noqa: E402
from bmr.rotor.bemt import BEMTRotor  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "data" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--charge-c-rate", type=float, default=0.5)
    args = parser.parse_args()

    # Build the mission demand profile (rotor + motor).
    airframe = Airframe.from_dict(load_yaml("aircraft.yaml"))
    rotor = BEMTRotor(load_yaml("rotor.yaml"), air_density_kg_m3=airframe.air_density_kg_m3)
    motor = PMSMMotor(load_yaml("motor.yaml"))
    mission = Mission.from_dict(load_yaml("mission_hover_cruise.yaml"))
    profile = build_demand_profile(mission, airframe, rotor, motor)

    # Force the degradation preset regardless of the file default.
    battery_yaml = load_yaml("battery.yaml")
    battery_yaml["preset"] = "life_degradation"
    config = BatteryConfig.from_dict(battery_yaml)

    print(f"Preset: life_degradation | parameter set: {config.parameter_set}")
    print(f"Cycles: {args.cycles} | charge: {args.charge_c_rate}C CCCV")
    print(f"Pack: {config.pack.series}s{config.pack.parallel}p")

    result = run_cycle_life(
        config, profile.time_s, profile.pack_power_W,
        n_cycles=args.cycles, charge_c_rate=args.charge_c_rate,
    )

    print("\n--- Capacity fade ---")
    for cyc, soh in zip(result.cycle_number, result.soh_percent):
        print(f"  cycle {int(cyc):3d}: SOH = {soh:6.3f} %")
    print(f"Final SOH: {result.final_soh_percent:.3f} %")
    if result.loss_breakdown_Ah:
        print("\n--- Capacity loss by mechanism (final cycle) ---")
        for key, vals in result.loss_breakdown_Ah.items():
            print(f"  {key}: {vals[-1]:.5f} A.h")

    columns = {
        "cycle_number": result.cycle_number,
        "capacity_Ah": result.capacity_Ah,
        "soh_percent": result.soh_percent,
    }
    for key, vals in result.loss_breakdown_Ah.items():
        columns[key] = vals
    save_system_csv(RESULTS / "degradation_study.csv", columns)
    print(f"\nWrote results to {RESULTS}")


if __name__ == "__main__":
    main()
