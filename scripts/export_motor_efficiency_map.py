#!/usr/bin/env python3
"""Export the PMSM torque-speed efficiency map (with field weakening) as a figure.

Sweeps a torque-speed grid at the nominal bus voltage, evaluating the
field-weakening control and voltage/current limits at each point, and writes a
contour map plus the underlying CSV.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmr.aircraft.airframe import Airframe  # noqa: E402
from bmr.config import load_yaml  # noqa: E402
from bmr.motor.calibration import calibrate_loss_coefficients  # noqa: E402
from bmr.motor.pmsm import PMSMMotor, compute_efficiency_map  # noqa: E402
from bmr.postprocess.plots import plot_efficiency_map, save_system_csv  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "data" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrate", action="store_true",
                        help="fit the loss coefficients to motor.yaml's "
                             "efficiency_targets before mapping")
    args = parser.parse_args()

    motor_cfg = load_yaml("motor.yaml")
    motor = PMSMMotor(motor_cfg)
    airframe = Airframe.from_dict(load_yaml("aircraft.yaml"))
    bus_v = airframe.nominal_bus_voltage_V
    tag = "raw"

    if args.calibrate:
        targets = motor_cfg["efficiency_targets"]
        points = [tuple(p) for p in targets["points"]]
        cal = calibrate_loss_coefficients(
            motor, points, targets.get("bus_voltage_V", bus_v))
        tag = "calibrated"
        print("--- Loss-coefficient calibration ---")
        print(f"Rs={cal.Rs:.4f} ohm | k_h={cal.k_h:.4f} | k_e={cal.k_e:.3e} | "
              f"c_mech={cal.c_mech:.3e}")
        print(f"Peak efficiency: {cal.peak_efficiency_before*100:.1f}% -> "
              f"{cal.peak_efficiency_after*100:.1f}%")
        print(f"Fit RMS error: {cal.rms_efficiency_error*100:.2f}% | "
              f"max: {cal.max_efficiency_error*100:.2f}%\n")

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
        RESULTS / f"motor_efficiency_map_{tag}.png", speed_rpm, torque_Nm, eff, fw,
        title=f"PMSM efficiency map ({tag}, bus = {bus_v:.0f} V, with field weakening)",
    )

    # Flatten to long-form CSV.
    S, T = np.meshgrid(speed_rpm, torque_Nm)
    save_system_csv(RESULTS / f"motor_efficiency_map_{tag}.csv", {
        "speed_rpm": S.ravel(),
        "torque_Nm": T.ravel(),
        "efficiency": eff.ravel(),
        "field_weakening": fw.ravel().astype(int),
    })
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
