#!/usr/bin/env python3
"""Stage 2: rotor (BEMT) + motor (PMSM) -> battery power demand profile.

Builds the propulsion demand for the mission without solving the battery, and
prints hover sanity checks (thrust vs weight, figure of merit, motor efficiency).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmr.aircraft.airframe import Airframe  # noqa: E402
from bmr.config import load_yaml  # noqa: E402
from bmr.coupling.profile_builder import build_demand_profile  # noqa: E402
from bmr.mission.profile import Mission  # noqa: E402
from bmr.motor.pmsm import PMSMMotor  # noqa: E402
from bmr.postprocess.plots import save_system_csv  # noqa: E402
from bmr.rotor.bemt import BEMTRotor  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "data" / "results"


def main() -> None:
    airframe = Airframe.from_dict(load_yaml("aircraft.yaml"))
    rotor = BEMTRotor(load_yaml("rotor.yaml"), air_density_kg_m3=airframe.air_density_kg_m3)
    motor = PMSMMotor(load_yaml("motor.yaml"))
    mission = Mission.from_dict(load_yaml("mission_hover_cruise.yaml"))

    # Hover sanity check.
    T_hover = airframe.per_rotor_thrust_N(1.0)
    hover = rotor.thrust_to_speed(T_hover)
    motor_hover = motor.operating_point(
        hover.torque_Nm, hover.rotor_speed_rad_s, airframe.nominal_bus_voltage_V)
    print("--- Hover operating point (per rotor) ---")
    print(f"Required thrust:   {T_hover:.1f} N (weight/{airframe.num_rotors})")
    print(f"Rotor speed:       {hover.rotor_speed_rad_s * 60 / (2 * np.pi):.0f} rpm")
    print(f"Rotor torque:      {hover.torque_Nm:.3f} N.m")
    print(f"Figure of merit:   {hover.figure_of_merit:.3f}")
    print(f"Motor efficiency:  {motor_hover.efficiency * 100:.1f} %")
    print(f"Pack power (hover):{motor_hover.dc_power_W * airframe.num_rotors / 1000:.3f} kW")

    profile = build_demand_profile(mission, airframe, rotor, motor)
    print("\n--- Mission demand profile ---")
    print(f"Duration:        {profile.time_s[-1] / 60:.1f} min")
    print(f"Peak pack power: {profile.pack_power_W.max() / 1000:.2f} kW")
    print(f"Mean pack power: {profile.pack_power_W.mean() / 1000:.2f} kW")

    save_system_csv(RESULTS / "demand_profile.csv", {
        "time_s": profile.time_s,
        "pack_power_W": profile.pack_power_W,
        "per_rotor_thrust_N": profile.per_rotor_thrust_N,
        "rotor_speed_rad_s": profile.rotor_speed_rad_s,
        "rotor_torque_Nm": profile.rotor_torque_Nm,
        "motor_efficiency": profile.motor_efficiency,
        "rotor_figure_of_merit": profile.rotor_figure_of_merit,
    })
    print(f"Wrote demand profile to {RESULTS}")


if __name__ == "__main__":
    main()
