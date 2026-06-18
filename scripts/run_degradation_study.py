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
from bmr.postprocess.plots import plot_soh_curve, save_system_csv  # noqa: E402
from bmr.rotor.bemt import BEMTRotor  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "data" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=60)
    parser.add_argument("--charge-c-rate", type=float, default=1.0)
    parser.add_argument("--full-profile", action="store_true",
                        help="discharge with the exact mission drive cycle each "
                             "cycle (much slower); default uses an energy-"
                             "equivalent constant-power discharge")
    args = parser.parse_args()

    # Build the mission demand profile (rotor + motor).
    airframe = Airframe.from_dict(load_yaml("aircraft.yaml"))
    rotor = BEMTRotor(load_yaml("rotor.yaml"), air_density_kg_m3=airframe.air_density_kg_m3)
    motor = PMSMMotor(load_yaml("motor.yaml"))
    mission = Mission.from_dict(load_yaml("mission_hover_cruise.yaml"))
    profile = build_demand_profile(mission, airframe, rotor, motor)

    time_s = profile.time_s
    if args.full_profile:
        pack_power = profile.pack_power_W
        discharge_desc = "full mission drive cycle"
    else:
        # Energy-equivalent constant-power discharge over the mission duration.
        mean_power = float(np.trapezoid(profile.pack_power_W, time_s)
                           / (time_s[-1] - time_s[0]))
        pack_power = np.full_like(time_s, mean_power)
        discharge_desc = f"constant {mean_power / 1000:.2f} kW (energy-equivalent)"

    # Force the degradation preset regardless of the file default.
    battery_yaml = load_yaml("battery.yaml")
    battery_yaml["preset"] = "life_degradation"
    config = BatteryConfig.from_dict(battery_yaml)

    print(f"Preset: life_degradation | parameter set: {config.parameter_set}")
    print(f"Cycles: {args.cycles} | charge: {args.charge_c_rate}C CCCV")
    print(f"Discharge: {discharge_desc}")
    print(f"Pack: {config.pack.series}s{config.pack.parallel}p")

    result = run_cycle_life(
        config, time_s, pack_power,
        n_cycles=args.cycles, charge_c_rate=args.charge_c_rate,
    )

    print("\n--- Capacity fade ---")
    n = len(result.cycle_number)
    stride = max(1, n // 10)  # print ~10 rows regardless of cycle count
    for idx in range(0, n, stride):
        cyc = int(result.cycle_number[idx])
        print(f"  cycle {cyc:3d}: SOH = {result.soh_percent[idx]:6.3f} %")
    print(f"Final SOH (cycle {int(result.cycle_number[-1])}): "
          f"{result.final_soh_percent:.3f} %")
    if result.loss_breakdown_Ah:
        print("\n--- Capacity loss by mechanism (final cycle) ---")
        for key, vals in result.loss_breakdown_Ah.items():
            print(f"  {key}: {vals[-1]:.5f} A.h")

    tag = "full" if args.full_profile else "representative"
    columns = {
        "cycle_number": result.cycle_number,
        "capacity_Ah": result.capacity_Ah,
        "soh_percent": result.soh_percent,
    }
    for key, vals in result.loss_breakdown_Ah.items():
        columns[key] = vals
    save_system_csv(RESULTS / f"degradation_study_{tag}.csv", columns)
    plot_soh_curve(
        RESULTS / f"degradation_soh_{tag}.png",
        result.cycle_number, result.soh_percent, result.loss_breakdown_Ah,
    )
    print(f"\nWrote results to {RESULTS} (tag: {tag})")


if __name__ == "__main__":
    main()
