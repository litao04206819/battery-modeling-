#!/usr/bin/env python3
"""Export the PMSM torque-speed efficiency map (with field weakening) as a figure.

Sweeps a torque-speed grid at the nominal bus voltage, evaluating the
field-weakening control and voltage/current limits at each point, and writes a
contour map plus the underlying CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmr.aircraft.airframe import Airframe  # noqa: E402
from bmr.config import load_yaml  # noqa: E402
from bmr.motor.pmsm import PMSMMotor, compute_efficiency_map  # noqa: E402
from bmr.postprocess.plots import plot_efficiency_map, save_system_csv  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "data" / "results"


def main() -> None:
    motor = PMSMMotor(load_yaml("motor.yaml"))
    airframe = Airframe.from_dict(load_yaml("aircraft.yaml"))
    bus_v = airframe.nominal_bus_voltage_V

    # No-load speed at the bus voltage sets the upper end (go a bit beyond to
    # show the field-weakening region).
    no_load_rpm = motor.KV * bus_v
    speed_rpm = np.linspace(200.0, 1.3 * no_load_rpm, 120)

    # Torque ceiling from the phase-current limit (Kt * I_max).
    tq_max = motor.Kt * motor.max_phase_current
    torque_Nm = np.linspace(0.02, tq_max, 100)

    print(f"Bus voltage: {bus_v:.1f} V | no-load speed: {no_load_rpm:.0f} rpm")
    print(f"Torque ceiling (Kt*Imax): {tq_max:.2f} N.m")

    eff, fw = compute_efficiency_map(motor, speed_rpm, torque_Nm, bus_v)

    finite = np.isfinite(eff)
    print(f"Feasible grid points: {finite.sum()} / {eff.size}")
    print(f"Peak efficiency: {np.nanmax(eff) * 100:.1f} %")
    print(f"Field-weakening points: {fw.sum()}")

    out_png = plot_efficiency_map(
        RESULTS / "motor_efficiency_map.png", speed_rpm, torque_Nm, eff, fw,
        title=f"PMSM efficiency map (bus = {bus_v:.0f} V, with field weakening)",
    )

    # Flatten to long-form CSV.
    S, T = np.meshgrid(speed_rpm, torque_Nm)
    save_system_csv(RESULTS / "motor_efficiency_map.csv", {
        "speed_rpm": S.ravel(),
        "torque_Nm": T.ravel(),
        "efficiency": eff.ravel(),
        "field_weakening": fw.ravel().astype(int),
    })
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
